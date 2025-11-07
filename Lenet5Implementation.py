import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
 
# Load MNIST dataset
transform = transforms.ToTensor()
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
 
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000)
 
# LeNet-5 architecture
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)    # (1, 28, 28) → (6, 24, 24)
        self.pool1 = nn.AvgPool2d(2)                   # (6, 24, 24) → (6, 12, 12)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)   # (6, 12, 12) → (16, 8, 8)
        self.pool2 = nn.AvgPool2d(2)                   # (16, 8, 8) → (16, 4, 4)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
 
    def forward(self, x):
        x = self.pool1(F.tanh(self.conv1(x)))
        x = self.pool2(F.tanh(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return self.fc3(x)
 
model = LeNet5()
 
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
 
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