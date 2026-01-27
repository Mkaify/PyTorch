import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import os

# Event classes (subset of UrbanSound8K)
EVENTS = ["dog_bark", "siren", "gun_shot", "engine_idling", "children_playing"]
EVENT2IDX = {ev: idx for idx, ev in enumerate(EVENTS)}

# Preprocessing pipeline
transform = nn.Sequential(
    MelSpectrogram(sample_rate=22050, n_mels=64),
    AmplitudeToDB()
)

# Custom dataset for Urban Sound Events
class UrbanEventDataset(Dataset):
    def __init__(self, root_dir):
        self.files = []
        self.labels = []
        for label in EVENTS:
            path = os.path.join(root_dir, label)
            for f in os.listdir(path):
                if f.endswith(".wav"):
                    self.files.append(os.path.join(path, f))
                    self.labels.append(EVENT2IDX[label])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.files[idx])
        waveform = waveform[:, :22050]  # 1 second
        mel = transform(waveform).squeeze(0)
        return mel, self.labels[idx]

# Load dataset
dataset = UrbanEventDataset("./data/urban_events")
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# CNN model
class EventClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x.unsqueeze(1))  # Add channel dimension

model = EventClassifier(num_classes=len(EVENTS))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(5):
    for x, y in loader:
        outputs = model(x)
        loss = criterion(outputs, torch.tensor(y))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")