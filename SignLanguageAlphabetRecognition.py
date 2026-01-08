import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
 
# Transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
 
# Load ASL Alphabet Dataset (e.g., ASL Alphabet dataset from Kaggle)
# Directory structure: data/asl_alphabet/A/, B/, ..., Z/
train_dataset = datasets.ImageFolder(root='./data/asl_alphabet/train', transform=transform)
test_dataset = datasets.ImageFolder(root='./data/asl_alphabet/test', transform=transform)
 
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)
 
# CNN Model
class SignLanguageCNN(nn.Module):
    def __init__(self, num_classes=26):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
 
    def forward(self, x):
        return self.net(x)
 
model = SignLanguageCNN(num_classes=26)
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
 
# Evaluation
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
 
print(f"Test Accuracy: {100 * correct / total:.2f}%")