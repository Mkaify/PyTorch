import torch
import torch.nn as nn
import torch.nn.functional as F
 
# Define a simple model
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)
 
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
 
# Instantiate and train model
model = SimpleMLP()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
 
# Dummy training data
X = torch.randn(100, 10)
Y = torch.randn(100, 1)
 
# Training loop
for epoch in range(10):
    outputs = model(X)
    loss = criterion(outputs, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
 
print("Training done.")
 
# Save model state_dict (recommended way)
torch.save(model.state_dict(), "simple_mlp.pth")
 
# To load later:
loaded_model = SimpleMLP()
loaded_model.load_state_dict(torch.load("simple_mlp.pth"))
loaded_model.eval()
 
# Optional: save the entire model object (less portable)
# torch.save(model, "simple_mlp_full.pth")
# model = torch.load("simple_mlp_full.pth")