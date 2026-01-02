import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
 
# Emotion labels (FER-2013 standard)
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
 
# Transform for grayscale face images
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
 
# Load FER2013 dataset (simulated with torchvision.datasets.ImageFolder here)
# In practice, you can convert FER2013 CSV into folders or use third-party wrappers
train_data = torchvision.datasets.ImageFolder(root="./data/faces/train", transform=transform)
test_data = torchvision.datasets.ImageFolder(root="./data/faces/test", transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=100)
 
# CNN model
class EmotionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 128),
            nn.ReLU(),
            nn.Linear(128, len(EMOTIONS))
        )
 
    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)
 
model = EmotionCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
 
# Train loop (short version)
for epoch in range(3):
    for x, y in train_loader:
        outputs = model(x)
        loss = criterion(outputs, y)
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
 
# Evaluate
correct, total = 0, 0
with torch.no_grad():
    for x, y in test_loader:
        outputs = model(x)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
 
print(f"Test Accuracy: {100 * correct / total:.2f}%")