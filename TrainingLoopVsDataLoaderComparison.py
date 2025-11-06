import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
 
# Load MNIST and apply transform
transform = transforms.ToTensor()
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
 
# Model for classification
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)
 
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
 
model1 = SimpleNN()  # For manual batching
model2 = SimpleNN()  # For DataLoader
 
# Shared loss and optimizers
criterion = nn.CrossEntropyLoss()
optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.01)
optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)
 
# -------- Manual Batching --------
X = train_data.data.view(-1, 1, 28, 28).float() / 255.0
Y = train_data.targets
 
batch_size = 64
for epoch in range(1):
    for i in range(0, len(X), batch_size):
        x_batch = X[i:i+batch_size]
        y_batch = Y[i:i+batch_size]
        outputs = model1(x_batch)
        loss = criterion(outputs, y_batch)
 
        optimizer1.zero_grad()
        loss.backward()
        optimizer1.step()
print("Manual batching epoch done.")
 
# -------- DataLoader Batching --------
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
 
for epoch in range(1):
    for images, labels in train_loader:
        outputs = model2(images)
        loss = criterion(outputs, labels)
 
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()
print("DataLoader batching epoch done.")