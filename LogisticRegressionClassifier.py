import torch
import torch.nn as nn

# Example input features and binary labels (0 or 1)
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])  # Feature values
Y = torch.tensor([[0.0], [0.0], [1.0], [1.0]])  # Labels

# Define logistic regression model (linear layer followed by sigmoid)
class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # One input and one output neuron

    def forward(self, x):
        return torch.sigmoid(self.linear(x))  # Apply sigmoid to output

model = LogisticRegression()  # Instantiate the model

# Binary Cross-Entropy loss for binary classification
criterion = nn.BCELoss()

# Stochastic Gradient Descent optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Training loop
for epoch in range(1000):
    y_pred = model(X)  # Forward pass

    loss = criterion(y_pred, Y)  # Compute binary cross-entropy loss

    optimizer.zero_grad()  # Reset gradients

    loss.backward()  # Backpropagation

    optimizer.step()  # Update weights

    # Print every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# Print the learned parameters
params = list(model.parameters())
print(f'Learned weight: {params[0].item():.4f}, bias: {params[1].item():.4f}')