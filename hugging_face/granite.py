from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("instructlab/granite-7b-lab")

# Load the model
model = AutoModelForCausalLM.from_pretrained("instructlab/granite-7b-lab")

# Example text
input_text = "Hello, how are you?"

# Tokenize the input text
inputs = tokenizer(input_text, return_tensors="pt")

# Generate text
outputs = model.generate(**inputs, max_new_tokens=10)

# Decode the generated text
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(decoded_output)