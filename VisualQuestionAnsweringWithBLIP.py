from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import torch
 
# Load pretrained BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
 
# Load and prepare image
image = Image.open("data/dog_playing.jpg").convert("RGB")  # Replace with your own image
 
# Define question
question = "What is the dog doing?"
 
# Preprocess inputs
inputs = processor(image, question, return_tensors="pt")
 
# Forward pass
with torch.no_grad():
    output = model.generate(**inputs)
 
# Decode and display answer
answer = processor.tokenizer.decode(output[0], skip_special_tokens=True)
print("Question:", question)
print("Answer:", answer)