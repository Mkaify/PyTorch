import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import os

# Simulated environment labels (replace with real dataset like DCASE or UrbanSound8K)
SCENES = ["street", "airport", "train", "shopping_mall", "cafe"]
SCENE2IDX = {scene: idx for idx, scene in enumerate(SCENES)}

# Preprocessing pipeline
transform = nn.Sequential(
    MelSpectrogram(sample_rate=16000, n_mels=64),
    AmplitudeToDB()
)

# Custom dataset class
class SceneDataset(Dataset):
    def __init__(self, root_dir):
        self.files = []
        self.labels = []
        for scene in SCENES:
            scene_dir = os.path.join(root_dir, scene)
            for file in os.listdir(scene_dir):
                if file.endswith(".wav"):
                    self.files.append(os.path.join(scene_dir, file))
                    self.labels.append(SCENE2IDX[scene])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.files[idx])
        waveform = waveform[:, :16000]  # use first 1s
        mel = transform(waveform).squeeze(0)
        return mel, self.labels[idx]

# Load dataset
dataset = SceneDataset("./data/scenes")
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# CNN classifier
class SceneClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel
        return self.net(x)

model = SceneClassifier(num_classes=len(SCENES))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train loop
for epoch in range(5):
    for x, y in loader:
        preds = model(x)
        loss = criterion(preds, torch.tensor(y))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")