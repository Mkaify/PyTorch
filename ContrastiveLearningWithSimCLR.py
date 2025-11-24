import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import random
 
# Define SimCLR augmentations
simclr_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
 
# Dataset that returns two augmented views of each image
class SimCLRDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
 
    def __getitem__(self, index):
        img, _ = self.dataset[index]
        return simclr_transform(img), simclr_transform(img)
 
    def __len__(self):
        return len(self.dataset)
 
# Load CIFAR-10
base_dataset = datasets.CIFAR10(root='./data', train=True, download=True)
train_dataset = SimCLRDataset(base_dataset)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
 
# SimCLR encoder with projection head
class SimCLR(nn.Module):
    def __init__(self):
        super(SimCLR, self).__init__()
        base = models.resnet18(pretrained=False)
        self.encoder = nn.Sequential(*list(base.children())[:-1])  # Remove classifier
        self.projector = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
 
    def forward(self, x):
        x = self.encoder(x).squeeze()
        x = self.projector(x)
        return F.normalize(x, dim=1)
 
# NT-Xent contrastive loss
def nt_xent_loss(z1, z2, temperature=0.5):
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # 2N x D
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)  # 2N x 2N
    sim = sim / temperature
 
    labels = torch.arange(N).repeat(2)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(z.device)
 
    # Mask out self-similarity
    mask = torch.eye(2*N, dtype=torch.bool).to(z.device)
    sim.masked_fill_(mask, -9e15)
 
    sim_exp = torch.exp(sim)
    sim_sum = sim_exp.sum(dim=1, keepdim=True)
    log_prob = sim - torch.log(sim_sum)
    loss = - (labels * log_prob).sum(dim=1) / labels.sum(dim=1)
    return loss.mean()
 
# Initialize model and optimizer
model = SimCLR()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
 
# Training loop (self-supervised)
for epoch in range(3):
    for x1, x2 in train_loader:
        z1 = model(x1)
        z2 = model(x2)
        loss = nt_xent_loss(z1, z2)
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
    print(f"Epoch {epoch+1}, Contrastive Loss: {loss.item():.4f}")