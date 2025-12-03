from transformers import T5Tokenizer, T5ForConditionalGeneration
 
# Load pretrained T5 model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")
 
# Input text to summarize
text = """
Artificial intelligence (AI) is intelligence demonstrated by machines, 
in contrast to the natural intelligence displayed by humans and animals. 
Leading AI textbooks define the field as the study of "intelligent agents": 
any device that perceives its environment and takes actions that maximize its chance 
of successfully achieving its goals.
"""
 
# Prepend task prefix required by T5
input_text = "summarize: " + text.strip()
inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
 
# Generate summary
summary_ids = model.generate(inputs, max_length=50, num_beams=4, early_stopping=True)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
 
print("Summary:")
print(summary)
