import torch
import torch.nn as nn
import torchaudio
from torchaudio.datasets import RAVDESS
from torchaudio.transforms import MFCC
from torch.utils.data import DataLoader
import os
 
# Load RAVDESS dataset (if available or simulate similar)
class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.dataset = RAVDESS(root=root, download=True)
        self.transform = transform
        self.label_map = {1: "neutral", 2: "calm", 3: "happy", 4: "sad", 5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"}
 
    def __getitem__(self, index):
        waveform, sample_rate, label, _, _, _, _, _ = self.dataset[index]
        mfcc = self.transform(waveform).squeeze(0).transpose(0, 1)  # [time, features]
        return mfcc, label - 1  # Zero-based label
 
    def __len__(self):
        return len(self.dataset)
 
# Feature extraction: MFCC
mfcc_transform = MFCC(sample_rate=16000, n_mfcc=40)
 
# Create dataset and dataloader
dataset = EmotionDataset(root="./data", transform=mfcc_transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)
 
# LSTM model for classification
class EmotionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
 
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n.squeeze(0))
 
model = EmotionLSTM(input_dim=40, hidden_dim=128, output_dim=8)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
 
# Train loop (1 epoch for demo)
for epoch in range(1):
    for x, y in dataloader:
        outputs = model(x)
        loss = criterion(outputs, y)
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")