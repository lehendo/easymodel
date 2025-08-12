#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

import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, pipeline
from peft import LoraConfig, get_peft_model
from huggingface_hub import login
import evaluate
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from polar.utils.text_analytics.gqs import compute_aggregate_gqs_from_finetuned_model
from polar.utils.text_analytics.tokenefficiency import (
    compute_summarization_efficiency,
    compute_qa_efficiency,
    compute_translation_efficiency,
    compute_paraphrasing_efficiency,
    compute_code_generation_efficiency
)
import torch


def detect_text_column(dataset):
    for column in dataset.column_names:
        if "text" in column.lower() or "sentence" in column.lower() or "content" in column.lower():
            return column
    for column in dataset.column_names:
        if isinstance(dataset[0][column], str):
            return column
    raise ValueError("No valid text column found. Please check the dataset.")


def tokenize_function(example, tokenizer, text_column, max_length):
    # Ensure the pad_token is set if it's not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token
        tokenizer.pad_token_id = tokenizer.eos_token_id  # Ensure the pad_token_id is set
        print(f"Pad token set to: {tokenizer.pad_token}")
        print(f"Pad token ID set to: {tokenizer.pad_token_id}")

    return tokenizer(example[text_column], padding="max_length", truncation=True, max_length=max_length)


def compute_perplexity(predictions, tokenizer):
    perplexity_metric = evaluate.load("perplexity")
    decoded_predictions = [tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]
    return perplexity_metric.compute(predictions=decoded_predictions)


def compute_semantic_similarity(predictions, references):
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    pred_embeddings = sentence_model.encode(predictions)
    ref_embeddings = sentence_model.encode(references)
    similarities = cosine_similarity(pred_embeddings, ref_embeddings)
    return similarities.diagonal().mean()


def compute_token_lengths(tokenizer, input_text, output_text):
    input_tokens = tokenizer.encode(input_text)
    output_tokens = tokenizer.encode(output_text)
    return len(input_tokens), len(output_tokens)


def run_analytics(model, tokenizer, datasets):
    print("\nRunning analytics...")

    # GQS Analytics
    print("\nComputing GQS metrics...")
    gqs_results = compute_aggregate_gqs_from_finetuned_model(model, datasets)
    print("GQS Results:", gqs_results)

    # Token Efficiency Analytics
    print("\nComputing Token Efficiency metrics...")
    for dataset in datasets:
        print(f"\nDataset: {dataset}")

        summarization_efficiency = compute_summarization_efficiency(model, dataset)
        print(f"Summarization Efficiency: {summarization_efficiency:.4f}")

        qa_efficiency = compute_qa_efficiency(model, dataset)
        print(f"Question Answering Efficiency: {qa_efficiency:.4f}")

        translation_efficiency = compute_translation_efficiency(model, dataset, "en", "fr")
        print(f"Translation Efficiency: {translation_efficiency:.4f}")

        paraphrasing_efficiency = compute_paraphrasing_efficiency(model, dataset)
        print(f"Paraphrasing Efficiency: {paraphrasing_efficiency:.4f}")

        code_generation_efficiency = compute_code_generation_efficiency(model, dataset)
        print(f"Code Generation Efficiency: {code_generation_efficiency:.4f}")


def finetune_model(base_model, datasets, output_space, api_key, num_labels=2, num_epochs=1, batch_size=8,
                   max_length=128, subset_size=1000):
    login(api_key)
    # tokenizer = AutoTokenizer.from_pretrained(base_model)
    # tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-4")
    # tokenizer.pad_token = "<|dummy_87|>"
    # tokenizer.padding_side = "left"
    from transformers import GPT2LMHeadModel

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    # if tokenizer.pad_token is None:
    #     if tokenizer.eos_token is None:
    #         tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #     else:
    #         tokenizer.pad_token = tokenizer.eos_token
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5,
                                                               torch_dtype="auto")
    # model = AutoModelForCausalLM.from_pretrained(base_model)
    # model = AutoModelForCausalLM.from_pretrained("microsoft/phi-4", torch_dtype=torch.bfloat16)
    model = GPT2LMHeadModel.from_pretrained(base_model)
    model.resize_token_embeddings(len(tokenizer))

    # model.resize_token_embeddings(len(tokenizer))

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        task_type="CAUSAL_LM",
        target_modules=["c_attn", "c_proj"]
    )

    model = get_peft_model(model, lora_config)

    tokenized_datasets = []
    for dataset_name in datasets:
        dataset = load_dataset(dataset_name, split="train")
        if subset_size:
            dataset = dataset.shuffle(seed=42).select(range(subset_size))
        text_column = detect_text_column(dataset)
        tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer, text_column, max_length),
                                        batched=True)
        tokenized_datasets.append(tokenized_dataset)

    # Handle validation split - fall back to 'test' if 'validation' doesn't exist
    combined_val_dataset = None
    for dataset_name in datasets:
        try:
            combined_val_dataset = load_dataset(dataset_name, split="validation")
        except KeyError:
            print(f"Validation split not found for {dataset_name}. Using test split instead.")
            combined_val_dataset = load_dataset(dataset_name, split="test")

        # Tokenize validation dataset
        text_column = detect_text_column(combined_val_dataset)
        combined_val_dataset = combined_val_dataset.map(lambda x: tokenize_function(x, tokenizer, text_column, max_length),
                                                    batched=True)

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        push_to_hub=True,
        learning_rate=5e-5,
        gradient_accumulation_steps=4,
        logging_steps=500,
        bf16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets[0],
        eval_dataset=combined_val_dataset,
    )

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs} started...")
        trainer.train()

        print("Computing training perplexity...")
        train_predictions = trainer.predict(tokenized_datasets[0]).predictions
        train_perplexity = compute_perplexity(train_predictions, tokenizer)
        print(f"Training Perplexity (Epoch {epoch + 1}): {train_perplexity}")

        print("Computing validation perplexity...")
        val_predictions = trainer.predict(combined_val_dataset).predictions
        val_perplexity = compute_perplexity(val_predictions, tokenizer)
        print(f"Validation Perplexity (Epoch {epoch + 1}): {val_perplexity}")

        print("Computing semantic similarity...")
        val_references = [example[text_column] for example in combined_val_dataset]
        val_predictions = tokenizer.batch_decode(val_predictions, skip_special_tokens=True)
        semantic_similarity = compute_semantic_similarity(val_predictions, val_references)
        print(f"Semantic Similarity (Epoch {epoch + 1}): {semantic_similarity}")

    # Compute token lengths for each task
    tasks = ["summarization", "question_answering", "translation", "paraphrasing", "code_generation"]
    task_pipelines = {
        "summarization": pipeline("summarization", model=model, tokenizer=tokenizer),
        "question_answering": pipeline("question-answering", model=model, tokenizer=tokenizer),
        "translation": pipeline("translation", model=model, tokenizer=tokenizer),
        "paraphrasing": pipeline("text2text-generation", model=model, tokenizer=tokenizer),
        "code_generation": pipeline("text-generation", model=model, tokenizer=tokenizer)
    }

    for task in tasks:
        print(f"\nComputing token lengths for {task}...")
        input_lengths = []
        output_lengths = []

        for data in combined_val_dataset[:100]:  # Limit to 100 examples for efficiency
            input_text = data[text_column]

            if task == "summarization":
                output = task_pipelines[task](input_text, max_length=100, min_length=30, do_sample=False)[0][
                    'summary_text']
            elif task == "question_answering":
                output = task_pipelines[task](question=input_text, context=input_text)['answer']
            elif task == "translation":
                output = task_pipelines[task](input_text, max_length=100)[0]['translation_text']
            elif task == "paraphrasing":
                output = task_pipelines[task](f"Paraphrase: {input_text}", max_length=100, do_sample=True)[0][
                    'generated_text']
            else:  # code_generation
                output = task_pipelines[task](input_text, max_length=200, do_sample=True)[0]['generated_text']

            input_length, output_length = compute_token_lengths(tokenizer, input_text, output)
            input_lengths.append(input_length)
            output_lengths.append(output_length)

        avg_input_length = sum(input_lengths) / len(input_lengths)
        avg_output_length = sum(output_lengths) / len(output_lengths)
        print(
            f"{task.capitalize()} - Avg Input Length: {avg_input_length:.2f}, Avg Output Length: {avg_output_length:.2f}")

    run_analytics(model, tokenizer, datasets)

    model.push_to_hub(output_space)
    print(f"Model fine-tuned and pushed to Hugging Face Hub under space: {output_space}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Hugging Face models with specified datasets.")
    parser.add_argument("--base_model", type=str, required=True, help="Name of the base Hugging Face model.")
    parser.add_argument("--datasets", type=str, required=True, nargs="+", help="List of Hugging Face dataset names.")
    parser.add_argument("--output_space", type=str, required=True,
                        help="Hugging Face Hub repo name for the fine-tuned model.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length for tokenization.")
    parser.add_argument("--subset_size", type=int, default=1000, help="Subset size of the training dataset.")
    parser.add_argument("--api_key", type=str,
                        help="Hugging Face API key (optional, will be prompted if not provided).")
    args = parser.parse_args()

    api_key = args.api_key
    if not api_key:
        api_key = input("Please enter your Hugging Face API key: ")
        if not api_key:
            raise ValueError("Hugging Face API key is required to push the model to the hub.")

    finetune_model(
        base_model=args.base_model,
        datasets=args.datasets,
        output_space=args.output_space,
        api_key=api_key,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        subset_size=args.subset_size
    )
