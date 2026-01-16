import torch
import torch.nn as nn
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torch.utils.data import Dataset, DataLoader
import random

# Preprocessing pipeline
mel_transform = nn.Sequential(
    MelSpectrogram(sample_rate=16000, n_mels=64),
    AmplitudeToDB()
)

# Dataset of paired noisy and clean audio
class NoisyAudioDataset(Dataset):
    def __init__(self, audio_dir, noise_factor=0.05):
        self.files = [f"{audio_dir}/{f}" for f in os.listdir(audio_dir) if f.endswith(".wav")]
        self.noise_factor = noise_factor

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.files[idx])
        waveform = waveform[:, :16000]  # 1 second
        clean_mel = mel_transform(waveform).squeeze(0)

        # Add noise
        noise = torch.randn_like(waveform) * self.noise_factor
        noisy_waveform = waveform + noise
        noisy_mel = mel_transform(noisy_waveform).squeeze(0)

        return noisy_mel, clean_mel

# Autoencoder model
class AudioAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64 * 126, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 64 * 126),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.view(x.size(0), 64, 126)

# Load dataset
dataset = NoisyAudioDataset("./data/clean_audio")
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Train setup
model = AudioAutoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(5):
    for noisy, clean in loader:
        output = model(noisy)
        loss = criterion(output, clean)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")