import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
 
# Add noise to MNIST images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x + 0.5 * torch.randn_like(x)),  # Add Gaussian noise
    transforms.Lambda(lambda x: torch.clamp(x, 0., 1.))          # Clip to [0, 1]
])
 
# Load noisy and original MNIST images
train_noisy = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_clean = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
 
noisy_loader = DataLoader(train_noisy, batch_size=128, shuffle=True)
clean_loader = DataLoader(train_clean, batch_size=128, shuffle=True)
 
# Zip noisy and clean loaders together
train_loader = zip(noisy_loader, clean_loader)
 
# Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.view(-1, 1, 28, 28)
 
model = Autoencoder()
 
# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
 
# Training loop
for epoch in range(5):
    for (noisy_imgs, _), (clean_imgs, _) in train_loader:
        outputs = model(noisy_imgs)
        loss = criterion(outputs, clean_imgs)
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")