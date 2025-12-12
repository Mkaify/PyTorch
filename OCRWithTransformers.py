from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
 
# Load pretrained TrOCR model and processor
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
 
# Load and convert image (should be a clean, high-contrast image of handwritten or printed text)
image = Image.open("data/handwritten_note.png").convert("RGB")  # Replace with your image
 
# Preprocess image
pixel_values = processor(images=image, return_tensors="pt").pixel_values
 
# Run model inference
with torch.no_grad():
    generated_ids = model.generate(pixel_values)
 
# Decode generated text
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("Recognized Text:\n", generated_text)