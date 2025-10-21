import torch
import torch.nn as nn

X = torch.tensor([[1.00], [2.00], [3.00],[4.00]])
Y = torch.tensor([[2.00], [4.00], [6.00], [8.00]])

# model
model = nn.Linear(in_features=1,out_features=1)
# loss function
criterion = nn.MSELoss()
# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
  y_pred = model(X)

  loss = criterion(y_pred, Y)

  optimizer.zero_grad()

  loss.backward()

  optimizer.step()

  if (epoch + 1) % 100==0:
    print(f"Epoch: {epoch + 1}, Loss: {loss.item():.4f}")

params = list(model.parameters())
print(f"Learned weights: {params[0].item()}, Learned biases:{params[1].item()}")