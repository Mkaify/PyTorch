import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import os
import numpy as np
 
# Sound labels (multi-label per clip)
LABELS = ["car", "dog", "speech", "sirens", "wind"]
LABEL2IDX = {label: i for i, label in enumerate(LABELS)}
 
# Transform
transform = nn.Sequential(
    MelSpectrogram(sample_rate=16000, n_mels=64),
    AmplitudeToDB()
)
 
# Dataset class for multi-label audio
class MultiLabelSoundDataset(Dataset):
    def __init__(self, root_dir, annotation_file):
        self.root = root_dir
        self.data = []
        with open(annotation_file, "r") as f:
            for line in f:
                file, tags = line.strip().split("\t")
                labels = tags.split(",")
                label_vec = np.zeros(len(LABELS), dtype=np.float32)
                for label in labels:
                    if label in LABEL2IDX:
                        label_vec[LABEL2IDX[label]] = 1.0
                self.data.append((file, label_vec))
 
    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, idx):
        file, label_vec = self.data[idx]
        waveform, sr = torchaudio.load(os.path.join(self.root, file))
        waveform = waveform[:, :16000]  # 1 second
        mel = transform(waveform).squeeze(0)
        return mel, torch.tensor(label_vec)
 
# Dummy annotations.txt format: (tab-separated)
# clip1.wav    dog,speech
# clip2.wav    car,sirens
dataset = MultiLabelSoundDataset("./data/env_clips", "./data/annotations.txt")
loader = DataLoader(dataset, batch_size=16, shuffle=True)
 
# Multi-label CNN classifier
class MultiLabelCNN(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(32, num_labels)
 
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel
        x = self.conv(x)
        return torch.sigmoid(self.fc(x.squeeze(-1).squeeze(-1)))
 
model = MultiLabelCNN(num_labels=len(LABELS))
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
 
# Training loop
for epoch in range(5):
    for x, y in loader:
        preds = model(x)
        loss = criterion(preds, y)
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")