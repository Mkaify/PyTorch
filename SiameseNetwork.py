import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import random
from PIL import Image
import os
 
# Sample transformation
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor()
])
 
# Dummy dataset using MNIST to simulate similarity
class SiameseMNIST(Dataset):
    def __init__(self, mnist_dataset):
        self.data = mnist_dataset
        self.transform = transform
 
    def __getitem__(self, index):
        img1, label1 = self.data[index]
        # Choose matching or non-matching image
        should_match = random.randint(0, 1)
        while True:
            img2, label2 = random.choice(self.data)
            if (label1 == label2 and should_match) or (label1 != label2 and not should_match):
                break
        return self.transform(img1), self.transform(img2), torch.tensor([int(label1 == label2)], dtype=torch.float)
 
    def __len__(self):
        return len(self.data)
 
# Base CNN for embedding
class SiameseCNN(nn.Module):
    def __init__(self):
        super(SiameseCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 22 * 22, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
 
    def forward_once(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
 
    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        return out1, out2
 
# Contrastive loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
 
    def forward(self, out1, out2, label):
        distance = F.pairwise_distance(out1, out2)
        loss = label * distance.pow(2) + (1 - label) * F.relu(self.margin - distance).pow(2)
        return loss.mean()
 
# Load MNIST and wrap in Siamese dataset
mnist = datasets.MNIST(root='./data', train=True, download=True)
siamese_dataset = SiameseMNIST(mnist)
loader = DataLoader(siamese_dataset, batch_size=32, shuffle=True)
 
# Model and training setup
model = SiameseCNN()
criterion = ContrastiveLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
 
# Train loop
for epoch in range(3):
    for img1, img2, label in loader:
        out1, out2 = model(img1, img2)
        loss = criterion(out1, out2, label)
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")