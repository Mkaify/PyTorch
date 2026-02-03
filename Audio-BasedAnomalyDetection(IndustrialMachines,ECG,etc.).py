import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import os
 
# Transform for audio
transform = nn.Sequential(
    MelSpectrogram(sample_rate=16000, n_mels=64),
    AmplitudeToDB()
)
 
# Simulated anomaly dataset
class AudioAnomalyDataset(Dataset):
    def __init__(self, root_dir):
        self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(".wav")]
 
    def __len__(self):
        return len(self.files)
 
    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.files[idx])
        waveform = waveform[:, :16000]  # 1 sec
        mel = transform(waveform).squeeze(0)
        return mel
 
# Load training data (normal samples only)
dataset = AudioAnomalyDataset("./data/normal_sounds")
loader = DataLoader(dataset, batch_size=8, shuffle=True)
 
# Simple autoencoder
class AnomalyAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 126, 256),
            nn.ReLU(),
            nn.Linear(256, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 64 * 126),
            nn.Unflatten(1, (64, 126))
        )
 
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
 
model = AnomalyAutoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
 
# Train on normal data only
for epoch in range(5):
    for x in loader:
        output = model(x)
        loss = criterion(output, x)
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
 
# Inference on new audio (normal or anomalous)
def detect_anomaly(audio_path, threshold=0.01):
    waveform, _ = torchaudio.load(audio_path)
    waveform = waveform[:, :16000]
    mel = transform(waveform).squeeze(0).unsqueeze(0)
    with torch.no_grad():
        recon = model(mel)
        error = torch.nn.functional.mse_loss(recon, mel).item()
    return "Anomaly Detected" if error > threshold else "Normal"
 
# Example test
print("🧠 Test result:", detect_anomaly("data/test/machine_anomaly.wav"))