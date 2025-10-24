import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# Define image transforms for preprocessing (resize, normalize)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize all images to 128x128
    transforms.ToTensor(),          # Convert images to tensor
    transforms.Normalize([0.5], [0.5])  # Normalize pixel values
])

# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("tongpython/cat-and-dog")

# print("Path to dataset files:", path)

# Assume folder structure: ./data/train/cat & ./data/train/dog
# Each folder contains corresponding images
train_dataset = datasets.ImageFolder(root='/kaggle/input/cat-and-dog/test_set', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define a simple CNN for binary classification
class CatDogCNN(nn.Module):
    def __init__(self):
        super(CatDogCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  # First convolutional layer
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)               # Downsampling
        self.fc1 = nn.Linear(32 * 32 * 32, 1)         # Fully connected layer for binary output

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 -> ReLU -> Pool
        x = x.view(-1, 32 * 32 * 32)          # Flatten before FC
        return torch.sigmoid(self.fc1(x))     # Sigmoid for binary output

model = CatDogCNN()

# Binary cross-entropy loss for binary classification
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(5):
    for images, labels in train_loader:
        labels = labels.float().unsqueeze(1)  # Convert to float and shape [B, 1]
        outputs = model(images)              # Forward pass
        loss = criterion(outputs, labels)    # Compute loss

        optimizer.zero_grad()                # Clear gradients
        loss.backward()                      # Backpropagation
        optimizer.step()                     # Update weights

    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')