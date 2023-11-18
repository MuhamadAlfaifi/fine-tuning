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
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
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
