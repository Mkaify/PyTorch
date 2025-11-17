import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
 
# Load pretrained model
model = models.resnet18(pretrained=True)
model.eval()
 
# Preprocess input image (use your own image if desired)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])
 
# Load and preprocess image
img_path = 'data/sample_dog.jpg'  # Replace with your image
img = Image.open(img_path).convert('RGB')
input_tensor = transform(img).unsqueeze(0)
 
# Hook to get gradients and activations
gradients = []
activations = []
 
def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])
 
def forward_hook(module, input, output):
    activations.append(output)
 
# Register hooks on the last convolutional layer
target_layer = model.layer4[1].conv2
target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)
 
# Forward and backward pass for a chosen class
output = model(input_tensor)
class_idx = output.argmax().item()
model.zero_grad()
output[0, class_idx].backward()
 
# Get hooked data
grads = gradients[0]
acts = activations[0].detach()
 
# Global average pooling of gradients
weights = grads.mean(dim=[2, 3], keepdim=True)
cam = (weights * acts).sum(dim=1).squeeze()
 
# Normalize CAM to [0, 1]
cam = cam.cpu().numpy()
cam = np.maximum(cam, 0)
cam = cam / cam.max()
 
# Resize CAM and overlay on original image
heatmap = cv2.resize(cam, (224, 224))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
 
original_img = cv2.cvtColor(np.array(img.resize((224, 224))), cv2.COLOR_RGB2BGR)
overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
 
# Display result
plt.figure(figsize=(8, 4))
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title("Grad-CAM Heatmap")
plt.axis('off')
plt.show()