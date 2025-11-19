import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
 
# Use GPU if available
device_cpu = torch.device("cpu")
device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# Data loading
transform = transforms.ToTensor()
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
 
# Define a simple MLP model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)
 
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
 
# Function to train the model on a given device
def train(device):
    model = SimpleNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
 
    start = time.time()
    for epoch in range(1):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
 
            outputs = model(images)
            loss = criterion(outputs, labels)
 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    end = time.time()
    print(f"Training on {device} took {end - start:.2f} seconds")
 
# Train on CPU and GPU
train(device_cpu)
train(device_gpu)