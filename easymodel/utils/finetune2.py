from transformers import (
   AutoTokenizer,
   AutoModelForSequenceClassification,
   AutoModelForSeq2SeqLM,
   AutoModelForCausalLM,
   Trainer,
   TrainingArguments,
   AutoModelForTokenClassification,
   TrainerCallback,
)
from datasets import load_dataset
from typing import Optional, Callable


class ProgressCallback(TrainerCallback):
    """Callback to track training progress and emit updates."""
    def __init__(self, progress_callback: Optional[Callable] = None, cancel_flag: Optional[Callable[[], bool]] = None):
        self.progress_callback = progress_callback
        self.cancel_flag = cancel_flag
        self.total_steps = 0
        self.current_step = 0
        self.current_epoch = 0
        self.total_epochs = 0
        
    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training."""
        if self.progress_callback:
            self.progress_callback({
                "stage": "initializing",
                "progress": 0,
                "message": "Initializing training..."
            })
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each epoch."""
        self.current_epoch = state.epoch + 1
        self.total_epochs = args.num_train_epochs
        if self.progress_callback:
            self.progress_callback({
                "stage": "training",
                "progress": (self.current_epoch - 1) / self.total_epochs * 90,  # Reserve 10% for saving
                "epoch": self.current_epoch,
                "total_epochs": self.total_epochs,
                "message": f"Starting epoch {self.current_epoch}/{self.total_epochs}..."
            })
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logs are written."""
        # Check for cancellation
        if self.cancel_flag and self.cancel_flag():
            control.should_training_stop = True
            if self.progress_callback:
                self.progress_callback({
                    "stage": "cancelled",
                    "progress": 0,
                    "message": "Training cancelled by user"
                })
            return
        
        if self.progress_callback and state.global_step > 0:
            # Calculate progress within current epoch
            epoch_progress = state.global_step / state.max_steps if state.max_steps > 0 else 0
            # Overall progress: (epoch-1)/total_epochs + (epoch_progress/total_epochs) * 90%
            overall_progress = ((self.current_epoch - 1) / self.total_epochs + epoch_progress / self.total_epochs) * 90
            
            loss = logs.get("loss", None) if logs else None
            self.progress_callback({
                "stage": "training",
                "progress": overall_progress,
                "epoch": self.current_epoch,
                "total_epochs": self.total_epochs,
                "step": state.global_step,
                "loss": loss,
                "message": f"Epoch {self.current_epoch}/{self.total_epochs}, Step {state.global_step}/{state.max_steps}" + (f", Loss: {loss:.4f}" if loss else "")
            })
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of each epoch."""
        if self.progress_callback:
            self.progress_callback({
                "stage": "training",
                "progress": self.current_epoch / self.total_epochs * 90,
                "epoch": self.current_epoch,
                "total_epochs": self.total_epochs,
                "message": f"Completed epoch {self.current_epoch}/{self.total_epochs}"
            })
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        if self.progress_callback:
            self.progress_callback({
                "stage": "saving",
                "progress": 90,
                "message": "Training completed. Saving model..."
            })


def finetune_model(base_model, datasets, output_space, api_key, task_type="classification", text_field=None, label_field=None, num_epochs=1, batch_size=8, max_length=128, subset_size=1000, progress_callback: Optional[Callable] = None, cancel_flag: Optional[Callable[[], bool]] = None):
   """
   Fine-tune a Hugging Face model using PyTorch.


   Args:
       base_model (str): The model name or path.
       datasets (list[str]): A list of dataset names or paths.
       output_space (str): Hugging Face Space to save the model.
       api_key (str): Hugging Face API key for authentication.
       task_type (str): Type of task ('classification', 'seq2seq', 'generation', 'token_classification').
       text_field (str): The name of the text field in the dataset.
       label_field (str): The name of the label field in the dataset.
       num_epochs (int): Number of training epochs.
       batch_size (int): Batch size for training.
       max_length (int): Maximum token length for inputs.
       subset_size (int): Size of the dataset subset to use.
       progress_callback (Optional[Callable]): Optional callback function to receive progress updates.
       cancel_flag (Optional[Callable[[], bool]]): Optional function that returns True if training should be cancelled.
   """
   # Load the tokenizer and model dynamically based on the task
   tokenizer = AutoTokenizer.from_pretrained(base_model)
   
   # Fix tokenizer padding token if needed
   if tokenizer.pad_token is None:
       tokenizer.pad_token = tokenizer.eos_token


   if task_type == "classification":
       model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2)
   elif task_type == "seq2seq":
       model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
   elif task_type == "generation":
       model = AutoModelForCausalLM.from_pretrained(base_model)
   elif task_type == "token_classification":
       model = AutoModelForTokenClassification.from_pretrained(base_model)
   else:
       raise ValueError(f"Unsupported task_type: {task_type}")


   # Load datasets
   raw_datasets = load_dataset(datasets[0])  # Load the first dataset


   # Auto-detect splits and check for presence of 'train', 'validation', and 'test'
   train_data = raw_datasets.get('train', None)
   val_data = raw_datasets.get('validation', None)
   test_data = raw_datasets.get('test', None)


   # Handle subset if necessary
   if train_data:
       train_data = train_data.select(range(min(subset_size, len(train_data))))
   if val_data:
       val_data = val_data.select(range(min(subset_size, len(val_data))))
   if test_data:
       test_data = test_data.select(range(min(subset_size, len(test_data))))


   # Auto-detect text and label fields if not provided
   if not text_field:
       text_field = next((field for field in train_data.column_names if "text" in field.lower()), None)
       if not text_field:
           raise ValueError("Text field could not be auto-detected. Please specify `text_field`.")


   if not label_field and task_type in ["classification", "token_classification"]:
       label_field = next((field for field in train_data.column_names if "label" in field.lower()), None)
       if not label_field:
           raise ValueError("Label field could not be auto-detected. Please specify `label_field`.")


   # Tokenization functions
   def preprocess_with_labels(examples):
       tokenized = tokenizer(
           examples[text_field],
           truncation=True,
           padding="max_length",
           max_length=max_length,
           return_tensors="pt"
       )
       if label_field:
           tokenized["labels"] = examples[label_field]
       return tokenized


   preprocess_map = {
       "classification": preprocess_with_labels,
       "seq2seq": preprocess_with_labels,
       "generation": preprocess_with_labels,
       "token_classification": preprocess_with_labels,
   }


   # Tokenize datasets
   tokenized_train = train_data.map(preprocess_map[task_type], batched=True)
   tokenized_val = val_data.map(preprocess_map[task_type], batched=True) if val_data else None


   # Format datasets for PyTorch
   tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
   if tokenized_val:
       tokenized_val.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])


   # Set up training arguments
   training_args = TrainingArguments(
       output_dir="./results",
       num_train_epochs=num_epochs,
       per_device_train_batch_size=batch_size,
       save_strategy="epoch",
       save_total_limit=1,
       logging_dir="./logs",
       logging_steps=10,
       push_to_hub=True,
       hub_model_id=output_space,
       hub_token=api_key
   )


   # Set up callbacks
   callbacks = []
   if progress_callback:
       callbacks.append(ProgressCallback(progress_callback=progress_callback, cancel_flag=cancel_flag))

   # Set up trainer
   trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=tokenized_train,
       eval_dataset=tokenized_val,
       tokenizer=tokenizer,
       callbacks=callbacks
   )

   # Train the model
   trainer.train()

   # Emit progress update for saving
   if progress_callback:
       progress_callback({
           "stage": "saving",
           "progress": 95,
           "message": "Pushing model to Hugging Face Hub..."
       })

   # Save the model to Hugging Face Hub
   trainer.push_to_hub()
   
   # Emit final progress update
   if progress_callback:
       progress_callback({
           "stage": "completed",
           "progress": 100,
           "message": "Fine-tuning completed successfully!"
       })
