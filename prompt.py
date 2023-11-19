from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the fine-tuned model and tokenizer
model_path = "/Users/muhamad/Code/Play/ai-fine-tuning/results.html"  # Adjust path as necessary
# model_path = "gpt2-medium"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

tokenizer.pad_token = tokenizer.eos_token

def generate_text(prompt, max_length=50):
    # Ensure the pad_token is set for the tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    # Encode the prompt text to tensor
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=max_length)

    # Generate a sequence of text from the prompt
    output = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=1
    )

    # Decode and print the generated sequence
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Example prompt
prompt = "System: HTML Chat Assistant that only speaks in html tags\nUser: Hello is a nice word.\nAssistant: ```"
generated_text = generate_text(prompt)
print(generated_text)
