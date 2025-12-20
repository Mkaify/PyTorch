import torch
import torch.nn as nn
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.transforms import MelSpectrogram
from torch.utils.data import DataLoader
import os
 
# Target keywords
KEYWORDS = ["yes", "no", "stop"]
LABEL2IDX = {kw: idx for idx, kw in enumerate(KEYWORDS)}
 
# Subset loader for specific keywords
class KeywordDataset(SPEECHCOMMANDS):
    def __init__(self, subset):
        super().__init__("./data", download=True)
        self._walker = self._load_list(subset)
 
    def _load_list(self, subset):
        with open(os.path.join(self._path, f"{subset}_list.txt")) as f:
            files = [line.strip() for line in f]
        return [os.path.join(self._path, f) for f in files if any(kw in f for kw in KEYWORDS)]
 
# Preprocessing
mel_transform = MelSpectrogram(sample_rate=16000, n_mels=64)
 
def collate_fn(batch):
    inputs, targets = [], []
    for waveform, _, label, *_ in batch:
        if label in LABEL2IDX:
            mel = mel_transform(waveform).squeeze(0)
            inputs.append(mel)
            targets.append(LABEL2IDX[label])
    return torch.stack(inputs), torch.tensor(targets)
 
# Load dataset
trainset = KeywordDataset("training")
train_loader = DataLoader(trainset, batch_size=16, shuffle=True, collate_fn=collate_fn)
 
# CNN for classification
class KeywordCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(32, len(KEYWORDS))
 
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel
        x = self.conv(x)
        return self.fc(x.squeeze(-1).squeeze(-1))
 
model = KeywordCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
 
# Train loop
for epoch in range(5):
    for x, y in train_loader:
        outputs = model(x)
        loss = criterion(outputs, y)
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")