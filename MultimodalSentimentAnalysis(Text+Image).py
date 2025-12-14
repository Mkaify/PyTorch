from transformers import ViltProcessor, ViltForImageAndTextClassification
from PIL import Image
import torch
 
# Load pretrained ViLT model and processor
model = ViltForImageAndTextClassification.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
 
# Replace with your own image and caption
image = Image.open("data/smile_selfie.jpg").convert("RGB")
text = "I'm feeling amazing today!"
 
# Preprocess multimodal input
inputs = processor(text=[text], images=image, return_tensors="pt")
 
# Forward pass
with torch.no_grad():
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)
 
# Dummy labels (ViLT was originally trained for VQA tasks)
# You can fine-tune it for sentiment classification using your own dataset
print("Logits:", logits)
print("Probabilities:", probs)