import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Load FashionMNIST dataset
transform = transforms.ToTensor()
train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# Define MLP with BatchNorm
class MLPWithBatchNorm(nn.Module):
    def __init__(self):
        super(MLPWithBatchNorm, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.bn1 = nn.BatchNorm1d(256)           # BatchNorm after first layer
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)           # BatchNorm after second layer
        self.out = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)                    # Flatten image
        x = F.relu(self.bn1(self.fc1(x)))        # Linear -> BatchNorm -> ReLU
        x = F.relu(self.bn2(self.fc2(x)))        # Repeat for second layer
        return self.out(x)                       # Output layer

model = MLPWithBatchNorm()

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