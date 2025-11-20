import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
 
# Load MNIST
transform = transforms.ToTensor()
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
 
# Define a simple MLP
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)
 
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
 
# Function to train and return loss list
def train_model(optimizer_type):
    model = MLP()
    criterion = nn.CrossEntropyLoss()
    if optimizer_type == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
 
    losses = []
    for epoch in range(5):
        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    return losses
 
# Train with both optimizers
start = time.time()
sgd_losses = train_model("SGD")
sgd_time = time.time() - start
 
start = time.time()
adam_losses = train_model("Adam")
adam_time = time.time() - start
 
# Plotting
plt.plot(sgd_losses, label=f"SGD ({sgd_time:.2f}s)")
plt.plot(adam_losses, label=f"Adam ({adam_time:.2f}s)")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.title("SGD vs Adam - Training Loss")
plt.legend()
plt.show()