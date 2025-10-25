import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Download FashionMNIST dataset and apply transforms
transform = transforms.ToTensor()  # Convert images to tensor
train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Create data loaders
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000)

# Define a softmax classifier (no hidden layers, just input to output)
class SoftmaxClassifier(nn.Module):
    def __init__(self):
        super(SoftmaxClassifier, self).__init__()
        self.linear = nn.Linear(28*28, 10)  # 28x28 input -> 10 classes

    def forward(self, x):
        x = x.view(-1, 28*28)              # Flatten image
        return self.linear(x)              # Output raw logits (softmax in loss)

model = SoftmaxClassifier()

# Cross-entropy loss applies softmax internally
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Training loop
for epoch in range(5):
    for images, labels in train_loader:
        outputs = model(images)              # Forward pass
        loss = criterion(outputs, labels)    # Compute loss

        optimizer.zero_grad()                # Clear gradients
        loss.backward()                      # Backpropagation
        optimizer.step()                     # Update weights

    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# Evaluate accuracy on test set
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  # Choose class with max logit
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')