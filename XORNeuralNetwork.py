import torch
import torch.nn as nn
import torch.nn.functional as F
 
# XOR input and output data
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
Y = torch.tensor([[0.], [1.], [1.], [0.]])
 
# Define a small feedforward neural network
class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.fc1 = nn.Linear(2, 4)   # Hidden layer with 4 neurons
        self.fc2 = nn.Linear(4, 1)   # Output layer with 1 neuron
 
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))  # Non-linear activation (sigmoid)
        return torch.sigmoid(self.fc2(x))  # Final output (sigmoid for binary)
 
model = XORNet()
 
# Binary cross-entropy loss
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
 
# Training loop
for epoch in range(5000):
    outputs = model(X)
    loss = criterion(outputs, Y)
 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
 
    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
 
# Print final predictions
with torch.no_grad():
    predictions = model(X).round()
    print("\nPredictions:\n", predictions)