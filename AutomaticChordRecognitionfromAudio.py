import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import os

# Chord label map
CHORDS = ["C", "Dm", "Em", "F", "G", "Am", "Bdim"]
CHORD_TO_IDX = {chord: i for i, chord in enumerate(CHORDS)}

# Feature extraction: Chroma STFT
def extract_chroma(file_path, sr=22050):
    y, _ = librosa.load(file_path, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)  # 12 chroma bins
    return torch.tensor(chroma_mean, dtype=torch.float32)

# Custom chord dataset
class ChordDataset(Dataset):
    def __init__(self, data_dir):
        self.files = []
        self.labels = []
        for label in CHORDS:
            chord_dir = os.path.join(data_dir, label)
            for f in os.listdir(chord_dir):
                if f.endswith(".wav"):
                    self.files.append(os.path.join(chord_dir, f))
                    self.labels.append(CHORD_TO_IDX[label])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = extract_chroma(self.files[idx])
        y = self.labels[idx]
        return x, y

# Load dataset (e.g., pre-segmented chord .wav files)
dataset = ChordDataset("./data/chords")
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Simple MLP model for classification
class ChordClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(12, 32),
            nn.ReLU(),
            nn.Linear(32, len(CHORDS))
        )

    def forward(self, x):
        return self.model(x)

model = ChordClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train loop
for epoch in range(5):
    for x, y in loader:
        outputs = model(x)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Inference example
test_file = "./data/chords/G/test_g_chord.wav"
test_feat = extract_chroma(test_file).unsqueeze(0)
pred = model(test_feat)
chord = CHORDS[torch.argmax(pred).item()]
print("🎵 Predicted Chord:", chord)