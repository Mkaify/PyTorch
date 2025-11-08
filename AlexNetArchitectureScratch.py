import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
 
# Resize CIFAR-10 to 224x224 for AlexNet input
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
 
# Load dataset
train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
 
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000)
 
# AlexNet model
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  # (64, 55, 55)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # (64, 27, 27)
 
            nn.Conv2d(64, 192, kernel_size=5, padding=2),           # (192, 27, 27)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # (192, 13, 13)
 
            nn.Conv2d(192, 384, kernel_size=3, padding=1),          # (384, 13, 13)
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),          # (256, 13, 13)
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),          # (256, 13, 13)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)                   # (256, 6, 6)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 10)  # CIFAR-10 has 10 classes
        )
 
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 256 * 6 * 6)
        return self.classifier(x)
 
model = AlexNet()
 
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
 
# Training loop
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