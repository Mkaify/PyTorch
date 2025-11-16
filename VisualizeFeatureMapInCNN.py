import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
 
# Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
 
# Load one batch from CIFAR-10
test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=1, shuffle=True)
 
# Simple CNN to extract features
class FeatureCNN(nn.Module):
    def __init__(self):
        super(FeatureCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32 * 8 * 8, 10)
 
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Feature maps we want
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        return self.fc(x)
 
model = FeatureCNN()
model.eval()
 
# Get a single image and label
image, label = next(iter(test_loader))
 
# Forward pass and extract feature maps from conv1
with torch.no_grad():
    feature_maps = F.relu(model.conv1(image))  # [1, 16, 32, 32]
 
# Plot first 6 feature maps
fig, axes = plt.subplots(1, 6, figsize=(15, 3))
for i in range(6):
    axes[i].imshow(feature_maps[0, i].cpu(), cmap='viridis')
    axes[i].axis('off')
plt.suptitle("Feature Maps from Conv1")
plt.show()