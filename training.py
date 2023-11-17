from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("distilgpt2")
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

dataset = load_dataset('text', data_files={'train': './dataset.txt'})

def tokenize_function(examples):
    # Set padding token if not already defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
  output_dir="./results",
  num_train_epochs=3,
  per_device_train_batch_size=2,
  warmup_steps=500,
  weight_decay=0.01,
)

trainer = Trainer(
  model=model,
  args=training_args,
  train_dataset=tokenized_datasets['train'],
)

trainer.train()