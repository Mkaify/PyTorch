from transformers import BertForQuestionAnswering, BertTokenizer
import torch
 
# Load pretrained model and tokenizer
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)
 
# Define context and question
context = """
PyTorch is an open source machine learning framework based on the Torch library, 
used for applications such as computer vision and natural language processing, 
primarily developed by Meta AI.
"""
question = "Who developed PyTorch?"
 
# Tokenize inputs
inputs = tokenizer.encode_plus(question, context, return_tensors="pt")
input_ids = inputs["input_ids"].tolist()[0]
 
# Predict start and end of answer
with torch.no_grad():
    outputs = model(**inputs)
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
 
# Get the most probable start and end token positions
start = torch.argmax(start_scores)
end = torch.argmax(end_scores)
 
# Decode answer tokens
answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[start:end+1]))
print("Question:", question)
print("Answer:", answer)