import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Generate some synthetic data
X = torch.linspace(-3, 3, 100).unsqueeze(1)  # 100 inputs from -3 to 3
Y = torch.sin(X) + 0.1 * torch.randn(X.size())  # Target: noisy sine wave

# Load data into a DataLoader
dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Define a custom activation function
def swish(x):
    return x * torch.sigmoid(x)  # Swish: x * sigmoid(x)

# Custom Swish activation as a module (for use in nn.Sequential)
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Define a simple feedforward neural network using the Swish activation
class SwishNet(nn.Module):
    def __init__(self):
        super(SwishNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),  # Input layer
            Swish(),           # Custom activation
            nn.Linear(32, 32),
            Swish(),           # Another Swish
            nn.Linear(32, 1)   # Output layer
        )

    def forward(self, x):
        return self.net(x)

model = SwishNet()

# Use Mean Squared Error loss
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(200):
    for batch_X, batch_Y in loader:
        output = model(batch_X)              # Forward pass
        loss = criterion(output, batch_Y)    # Compute MSE loss

        optimizer.zero_grad()                # Clear gradients
        loss.backward()                      # Backpropagation
        optimizer.step()                     # Update weights

    if (epoch+1) % 40 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')