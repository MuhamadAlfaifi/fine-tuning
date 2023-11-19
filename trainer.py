from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained('gpt2-medium')
tokenizer = AutoTokenizer.from_pretrained('gpt2-medium')

# Load and prepare the dataset
dataset = load_dataset('text', data_files={'train': './dataset/html_tags.txt'})

def tokenize_function(examples):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Tokenize the inputs and labels
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128, return_tensors="pt", return_attention_mask=True)

# Process the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Modify the dataset to include labels
tokenized_datasets = tokenized_datasets.map(lambda examples: {'labels': examples['input_ids']}, batched=True)


# Set training arguments
training_args = TrainingArguments(
    output_dir="results",
    num_train_epochs=3,  # Increase the number of epochs for more exposure to the training data
    per_device_train_batch_size=8,  # Slightly increase the batch size for potentially better learning
    warmup_steps=1000,  # Increase warmup steps for a more gradual learning rate increase at the start
    weight_decay=0.02,  # Increase weight decay for better regularization
    learning_rate=5e-5,  # Adjust the learning rate, 5e-5 is a common starting point for fine-tuning
    evaluation_strategy="no",  # Evaluate at the end of each epoch to monitor model performance
    save_strategy="epoch",  # Save the model at the end of each epoch
    logging_dir="logs",  # Log metrics during training for analysis
    logging_steps=50,  # Log metrics every 50 steps
)

# Initialize the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
)

# Start training
trainer.train()

model.save_pretrained('./results')
tokenizer.save_pretrained('./results')
