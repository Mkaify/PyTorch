import torch
import torch.nn as nn
import torch.nn.functional as F

# Generate simple synthetic data
X = torch.randn(10, 3)  # 10 samples, 3 features
Y = torch.randn(10, 1)  # 10 targets

# Define a small MLP model
class SmallNet(nn.Module):
    def __init__(self):
        super(SmallNet, self).__init__()
        self.fc1 = nn.Linear(3, 5)  # Input -> Hidden
        self.fc2 = nn.Linear(5, 1)  # Hidden -> Output

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model = SmallNet()

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Define a hook function to capture gradients
def print_grad_hook(module, grad_input, grad_output):
    print(f"\n--- Gradient for {module.__class__.__name__} ---")
    print("Grad Input:", grad_input)
    print("Grad Output:", grad_output)

# Register a backward hook on the first layer
hook = model.fc1.register_full_backward_hook(print_grad_hook)

# Forward pass
output = model(X)
loss = criterion(output, Y)

# Backward pass (this triggers the hook)
optimizer.zero_grad()
loss.backward()
optimizer.step()

# Remove hook after inspection
hook.remove()