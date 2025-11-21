import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import copy
 
# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
 
def load_image(path):
    image = Image.open(path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image.to(torch.float)
 
# Load content and style images
content = load_image('data/content.jpg')
style = load_image('data/style.jpg')
 
# Load pretrained VGG-19 and freeze it
vgg = models.vgg19(pretrained=True).features.eval()
 
# Layers to extract style and content
content_layers = ['conv4_2']
style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
 
# Helper to get specific layers
def get_features(x, model, layers):
    features = {}
    i = 0
    for layer in model.children():
        x = layer(x)
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv{i}_1'
            if name in layers:
                features[name] = x
    return features
 
# Gram matrix for style representation
def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(c, h * w)
    G = torch.mm(features, features.t())
    return G / (c * h * w)
 
# Target image initialized as a copy of content image
target = content.clone().requires_grad_(True)
 
# Optimizer
optimizer = optim.Adam([target], lr=0.01)
style_weight = 1e5
content_weight = 1
 
# Extract features
style_features = get_features(style, vgg, style_layers)
content_features = get_features(content, vgg, content_layers)
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_layers}
 
# Optimization loop
for step in range(300):
    target_features = get_features(target, vgg, style_layers + content_layers)
    
    content_loss = F.mse_loss(target_features['conv4_2'], content_features['conv4_2'])
 
    style_loss = 0
    for layer in style_layers:
        target_gram = gram_matrix(target_features[layer])
        style_gram = style_grams[layer]
        style_loss += F.mse_loss(target_gram, style_gram)
 
    total_loss = content_weight * content_loss + style_weight * style_loss
 
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
 
    if (step + 1) % 50 == 0:
        print(f"Step {step+1}, Loss: {total_loss.item():.4f}")
 
# Display final stylized image
output = target.detach().squeeze().permute(1, 2, 0).clamp(0, 1).numpy()
plt.imshow(output)
plt.title("Stylized Image")
plt.axis('off')
plt.show()