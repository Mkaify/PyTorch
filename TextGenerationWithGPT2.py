from transformers import GPT2Tokenizer, GPT2LMHeadModel
 
# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()
 
# Input prompt
prompt = "Once upon a time in a distant galaxy"
 
# Encode prompt text
input_ids = tokenizer.encode(prompt, return_tensors="pt")
 
# Generate text
output_ids = model.generate(
    input_ids,
    max_length=100,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.9
)
 
# Decode and print
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Generated Text:\n")
print(generated_text)