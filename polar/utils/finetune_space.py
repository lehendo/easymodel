import argparse
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model
from huggingface_hub import login, create_repo, upload_file

# Define the function for fine-tuning and pushing to Space
def finetune_and_deploy(base_model, datasets, output_space, api_key, num_labels=2, num_epochs=1, batch_size=8, max_length=128, subset_size=1000):
    # Login to Hugging Face Hub
    login(api_key)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=num_labels)

    # Apply LoRA for lightweight fine-tuning
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        task_type="SEQ_CLS"
    )
    model = get_peft_model(model, lora_config)

    # Load and tokenize datasets
    tokenized_datasets = []
    for dataset_name in datasets:
        dataset = load_dataset(dataset_name, split="train")
        
        # Reduce dataset size
        if subset_size:
            dataset = dataset.shuffle(seed=42).select(range(subset_size))

        tokenized_dataset = dataset.map(
            lambda x: tokenizer(x['text'], truncation=True, padding="max_length", max_length=max_length),
            batched=True
        )
        tokenized_datasets.append(tokenized_dataset)

    # Combine all datasets
    combined_dataset = concatenate_datasets(tokenized_datasets)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_epochs,
        evaluation_strategy="no",
        save_strategy="epoch",
        logging_dir="./logs",
        push_to_hub=False,  # Push manually to Space instead
        learning_rate=5e-5,
        gradient_accumulation_steps=4,
        logging_steps=500,
        bf16=True
    )

    # Create a Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=combined_dataset,
    )

    # Train the model
    trainer.train()

    # Save model locally
    model_dir = "./fine_tuned_model"
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    # Create Space repository with SDK
    print(f"Creating Hugging Face Space: {output_space}")
    create_repo(repo_id=output_space, repo_type="space", space_sdk="gradio", exist_ok=True)

    # Upload model files and app script to the Space
    files = {
        "app.py": """
import gradio as gr
from transformers import pipeline

# Load the model
model = pipeline("text-classification", model=".")

def classify_text(text):
    return model(text)

# Create the Gradio app
iface = gr.Interface(fn=classify_text, inputs="text", outputs="label", title="Text Classification")
iface.launch()
        """,
        "requirements.txt": "transformers\ngradio"
    }

    for filename, content in files.items():
        with open(filename, "w") as f:
            f.write(content)

    for filename in ["app.py", "requirements.txt"]:
        upload_file(
            path_or_fileobj=filename,
            path_in_repo=filename,
            repo_id=output_space,
            repo_type="space",
        )

    for filename in os.listdir(model_dir):
        upload_file(
            path_or_fileobj=os.path.join(model_dir, filename),
            path_in_repo=filename,
            repo_id=output_space,
            repo_type="space",
        )

    print(f"Model and app deployed to Space: {output_space}")

# Main entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Hugging Face models and deploy to Spaces.")
    parser.add_argument("--base_model", type=str, required=True, help="Name of the base Hugging Face model.")
    parser.add_argument("--datasets", type=str, required=True, nargs="+", help="List of Hugging Face dataset names.")
    parser.add_argument("--output_space", type=str, required=True, help="Hugging Face Space repo name.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length for tokenization.")
    parser.add_argument("--subset_size", type=int, default=1000, help="Subset size of the training dataset.")
    parser.add_argument("--api_key", type=str, help="Hugging Face API key (optional, will be prompted if not provided).")
    args = parser.parse_args()

    # Prompt for the API key if not provided
    api_key = args.api_key
    if not api_key:
        api_key = input("Please enter your Hugging Face API key: ")
        if not api_key:
            raise ValueError("Hugging Face API key is required to push the model to the hub.")

    # Call the function
    finetune_and_deploy(
        base_model=args.base_model,
        datasets=args.datasets,
        output_space=args.output_space,
        api_key=api_key,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        subset_size=args.subset_size
    )
