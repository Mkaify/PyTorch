import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import random
 
# CIFAR-10 transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
 
# Load CIFAR-10
train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
 
# CNN model
class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 8 * 8, 10)
 
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        return self.fc(x)
 
model = BasicCNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
 
# MixUp function
def mixup_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
 
# CutMix function
def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size, _, h, w = x.size()
    index = torch.randperm(batch_size)
    cx, cy = np.random.randint(w), np.random.randint(h)
    cut_w, cut_h = int(w * np.sqrt(1 - lam)), int(h * np.sqrt(1 - lam))
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, w)
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, h)
    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
    return x, y, y[index], lam
 
# One training epoch with MixUp and CutMix
for epoch in range(3):
    for images, labels in train_loader:
        if random.random() < 0.5:
            inputs, targets_a, targets_b, lam = mixup_data(images, labels)
        else:
            inputs, targets_a, targets_b, lam = cutmix_data(images, labels)
 
        outputs = model(inputs)
        loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")