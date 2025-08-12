# from transformers import AutoTokenizer, Trainer
# from datasets import load_dataset
# import torch
# from sklearn.model_selection import train_test_split
# import evaluate
#
#
# perplexity_metric = evaluate.load("perplexity")
#
#
# # Function to prepare dataset
# def prepare_dataset(dataset_url, max_length=128, test_size=0.2, val_size=0.1):
#     dataset = load_dataset(dataset_url, split="train")
#
#     text_column = None
#     for column in dataset.column_names:
#         if "text" in column.lower():
#             text_column = column
#             break
#
#     if not text_column:
#         raise ValueError("No text column found in the dataset.")
#
#     # Split the dataset into train, validation, and test
#     train_data, temp_data = train_test_split(dataset, test_size=test_size + val_size, random_state=42)
#     val_data, test_data = train_test_split(temp_data, test_size=test_size / (val_size + test_size), random_state=42)
#
#     tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
#
#     train_data = train_data.map(
#         lambda x: tokenizer(x[text_column], padding="max_length", truncation=True, max_length=max_length), batched=True)
#     val_data = val_data.map(
#         lambda x: tokenizer(x[text_column], padding="max_length", truncation=True, max_length=max_length), batched=True)
#     test_data = test_data.map(
#         lambda x: tokenizer(x[text_column], padding="max_length", truncation=True, max_length=max_length), batched=True)
#
#     return train_data, val_data, test_data
#
#
# # Function to calculate perplexity
# def compute_perplexity(logits, labels):
#     predictions = torch.argmax(logits, dim=-1)
#
#     results = perplexity_metric.compute(predictions=predictions, references=labels)
#
#     return results["perplexity"]
#
#
# # Function to calculate perplexity during fine-tuning
# def compute_perplexity_during_finetuning(model, tokenizer, train_dataset, val_dataset, training_args, num_epochs=1):
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=val_dataset,
#     )
#
#     training_perplexity = []
#     validation_perplexity = []
#
#     for epoch in range(num_epochs):
#         trainer.train()
#
#         model.eval()
#         logits_train, labels_train = get_logits_and_labels(model, train_dataset, tokenizer)
#         train_perplexity = compute_perplexity(logits_train, labels_train)
#         training_perplexity.append(train_perplexity)
#
#         logits_val, labels_val = get_logits_and_labels(model, val_dataset, tokenizer)
#         val_perplexity = compute_perplexity(logits_val, labels_val)
#         validation_perplexity.append(val_perplexity)
#
#     return training_perplexity, validation_perplexity
#
#
# # Helper function to extract logits and labels
# def get_logits_and_labels(model, dataset, tokenizer, is_language_modeling=False):
#     input_ids = dataset["input_ids"]
#     attention_mask = dataset["attention_mask"]
#
#     if is_language_modeling:
#         labels = input_ids[:, 1:].contiguous()
#         input_ids = input_ids[:, :-1].contiguous()
#     else:
#         labels = dataset["labels"]
#
#     # Convert input ids to tensors
#     input_ids = torch.tensor(input_ids)
#     attention_mask = torch.tensor(attention_mask)
#     labels = torch.tensor(labels)
#
#     with torch.no_grad():
#         outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#     logits = outputs.logits
#     return logits, labels
#
#
# # Function to compute perplexity for a language model
# def compute_perplexity_lm(model, tokenizer, text, max_length=512):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
#     input_ids = inputs['input_ids']
#     attention_mask = inputs['attention_mask']
#
#     with torch.no_grad():
#         outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
#
#     loss = outputs.loss
#     perplexity = torch.exp(loss).item()
#
#     return perplexity
