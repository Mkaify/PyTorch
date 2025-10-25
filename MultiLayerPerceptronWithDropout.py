import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Load FashionMNIST and apply transforms
transform = transforms.ToTensor()
train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)

# Define a multi-layer perceptron with dropout layers
class MLPDropout(nn.Module):
    def __init__(self):
        super(MLPDropout, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)  # First hidden layer
        self.drop1 = nn.Dropout(0.5)      # Dropout after first layer
        self.fc2 = nn.Linear(256, 128)    # Second hidden layer
        self.drop2 = nn.Dropout(0.5)      # Dropout after second layer
        self.out = nn.Linear(128, 10)     # Output layer for 10 classes

    def forward(self, x):
        x = x.view(-1, 28*28)           # Flatten image
        x = F.relu(self.fc1(x))         # Apply ReLU
        x = self.drop1(x)               # Apply dropout
        x = F.relu(self.fc2(x))         # Apply ReLU
        x = self.drop2(x)               # Apply dropout
        return self.out(x)              # Output logits

model = MLPDropout()

# Use cross-entropy loss and Adam optimizer
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

    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# Evaluation on test data
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy with Dropout: {100 * correct / total:.2f}%')