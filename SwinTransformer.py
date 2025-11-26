import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import swin_t
 
# Preprocessing (resize to 224x224 for Swin)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
 
# CIFAR-10 dataset
train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000)
 
# Load pretrained Swin Transformer (tiny version)
model = swin_t(weights="DEFAULT")  # torchvision >= 0.13
model.head = nn.Linear(model.head.in_features, 10)  # CIFAR-10 has 10 classes
 
# Freeze all except head (optional)
for param in model.parameters():
    param.requires_grad = False
for param in model.head.parameters():
    param.requires_grad = True
 
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.head.parameters(), lr=0.001)
 
# Training loop (head only)
for epoch in range(3):
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