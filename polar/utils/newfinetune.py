# import numpy
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
# from datasets import load_dataset, DatasetDict
#
#
# def finetune_model(base_model, datasets, output_space, api_key, num_epochs=1, batch_size=8, max_length=128, subset_size=1000):
#     """
#     Fine-tune a Hugging Face model using PyTorch.
#
#     Args:
#         base_model (str): The model name or path.
#         datasets (list[str]): A list of dataset names or paths.
#         output_space (str): Hugging Face Space to save the model.
#         api_key (str): Hugging Face API key for authentication.
#         num_epochs (int): Number of training epochs.
#         batch_size (int): Batch size for training.
#         max_length (int): Maximum token length for inputs.
#         subset_size (int): Size of the dataset subset to use.
#     """
#     # Step 1: Load the tokenizer and model
#     tokenizer = AutoTokenizer.from_pretrained(base_model)
#     model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2)
#
#     # Step 2: Load datasets
#     dataset_name = datasets[0]  # For simplicity, only use the first dataset in the list
#     raw_datasets = load_dataset(dataset_name)
#
#     # Use a subset of the dataset
#     train_data = raw_datasets['train'].select(range(min(subset_size, len(raw_datasets['train']))))
#     val_data = raw_datasets['validation'].select(range(min(subset_size, len(raw_datasets['validation']))))
#
#     # Tokenize datasets
#     def preprocess(examples):
#         return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)
#
#     tokenized_train = train_data.map(preprocess, batched=True)
#     tokenized_val = val_data.map(preprocess, batched=True)
#
#     # Format datasets for PyTorch
#     tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
#     tokenized_val.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
#
#     # Step 3: Set up training arguments and Trainer
#     training_args = TrainingArguments(
#         output_dir="./results",
#         num_train_epochs=num_epochs,
#         per_device_train_batch_size=batch_size,
#         evaluation_strategy="epoch",
#         save_strategy="epoch",
#         save_total_limit=1,
#         logging_dir="./logs",
#         logging_steps=10,
#         push_to_hub=True,
#         hub_model_id=output_space,
#         hub_token=api_key
#     )
#
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=tokenized_train,
#         eval_dataset=tokenized_val,
#         tokenizer=tokenizer
#     )
#
#     # Step 4: Train the model
#     trainer.train()
#
#     # Step 5: Save the model to Hugging Face Hub
#     trainer.push_to_hub()
