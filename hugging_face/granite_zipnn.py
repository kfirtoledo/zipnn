from transformers import AutoModelForCausalLM, AutoTokenizer
from zipnn_ext import decompress_model
# Load the tokenizer
decompress_model("royleibov/granite-7b-instruct-ZipNN-Compressed")
tokenizer = AutoTokenizer.from_pretrained("royleibov/granite-7b-instruct-ZipNN-Compressed")

# Load the model
model = AutoModelForCausalLM.from_pretrained("royleibov/granite-7b-instruct-ZipNN-Compressed")

# Example text
input_text = "Hello, how are you?"

# Tokenize the input text
inputs = tokenizer(input_text, return_tensors="pt")

# Generate text
outputs = model.generate(**inputs, max_new_tokens=10)

# Decode the generated text
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(decoded_output)