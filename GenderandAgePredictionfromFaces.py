import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
 
# Transform for face images
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
 
# Simulated dataset (use real face+label dataset like UTKFace)
# Assumes folder structure: data/age_gender/<gender_age_class>/image.jpg
# For real use: create a dataset class that returns (image, gender_label, age_group_label)
dataset = datasets.ImageFolder(root="./data/age_gender", transform=transform)
loader = DataLoader(dataset, batch_size=16, shuffle=True)
 
# Shared CNN feature extractor
class AgeGenderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = models.resnet18(pretrained=True)
        self.features.fc = nn.Identity()  # remove final layer
 
        self.fc_gender = nn.Linear(512, 2)   # Binary gender classification
        self.fc_age = nn.Linear(512, 5)      # Age group classification: e.g., [0–20, 21–30, ...]
 
    def forward(self, x):
        x = self.features(x)
        gender_out = self.fc_gender(x)
        age_out = self.fc_age(x)
        return gender_out, age_out
 
model = AgeGenderModel()
criterion_gender = nn.CrossEntropyLoss()
criterion_age = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
 
# Train loop
for epoch in range(3):
    for x, y in loader:
        # Dummy: Treat even-odd folder index as gender, and range index as age
        gender_labels = y % 2
        age_labels = y // 2
 
        gender_pred, age_pred = model(x)
        loss_gender = criterion_gender(gender_pred, gender_labels)
        loss_age = criterion_age(age_pred, age_labels)
        loss = loss_gender + loss_age
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
    print(f"Epoch {epoch+1}, Total Loss: {loss.item():.4f}")