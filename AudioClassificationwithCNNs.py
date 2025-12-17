import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import os
 
# Preprocessing pipeline: Mel spectrogram + dB conversion
transform = nn.Sequential(
    MelSpectrogram(sample_rate=16000, n_mels=64),
    AmplitudeToDB()
)
 
# Custom dataset wrapper
class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset):
        super().__init__("./data", download=True)
        self.subset = subset
        self._walker = self._load_list(subset)
 
    def _load_list(self, subset):
        filepath = os.path.join(self._path, f"{subset}_list.txt")
        with open(filepath) as f:
            return [os.path.join(self._path, line.strip()) for line in f]
 
# Load subset of SpeechCommands (predefined .txt files)
dataset = SubsetSC("validation")
label_names = sorted(set(dat[2] for dat in dataset))
label_to_idx = {label: i for i, label in enumerate(label_names)}
 
# Collate function
def collate_fn(batch):
    waves, labels = [], []
    for waveform, _, label, *_ in batch:
        mel = transform(waveform).squeeze(0)
        waves.append(mel)
        labels.append(label_to_idx[label])
    return torch.stack(waves), torch.tensor(labels)
 
loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
 
# CNN model
class AudioCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
 
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.conv(x)
        return self.fc(x)
 
model = AudioCNN(num_classes=len(label_names))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
 
# Training loop (1 epoch for demo)
for epoch in range(1):
    for x, y in loader:
        outputs = model(x)
        loss = criterion(outputs, y)
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")