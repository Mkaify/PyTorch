import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
 
# Image preprocessing for EfficientNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # EfficientNet expects 224x224 input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                         std=[0.229, 0.224, 0.225])
])
 
# Load custom dataset (structure: root/class_name/images.jpg)
train_data = datasets.ImageFolder(root='./custom_data/train', transform=transform)
test_data = datasets.ImageFolder(root='./custom_data/test', transform=transform)
 
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)
 
# Load pretrained EfficientNet-B0
model = models.efficientnet_b0(pretrained=True)
 
# Replace final classification layer
num_classes = len(train_data.classes)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
 
# Freeze base layers
for param in model.features.parameters():
    param.requires_grad = False
 
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)
 
# Training loop
for epoch in range(5):
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
 
# Evaluation
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
 
print(f"Test Accuracy: {100 * correct / total:.2f}%")