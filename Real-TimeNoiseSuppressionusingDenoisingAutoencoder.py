import torch
import torchaudio
import sounddevice as sd
import numpy as np
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, InverseMelScale

# Parameters
SAMPLE_RATE = 16000
FRAME_DURATION = 1  # seconds
NUM_SAMPLES = SAMPLE_RATE * FRAME_DURATION

# Mel transforms
mel_spec = MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=64)
to_db = AmplitudeToDB()
inv_mel = InverseMelScale(n_stft=1025, n_mels=64)

# Load trained denoising autoencoder
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
            nn.Linear(512, 64 * 126)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.view(x.size(0), 64, 126)

model = AudioAutoencoder()
model.load_state_dict(torch.load("denoising_autoencoder.pth", map_location="cpu"))  # Replace with your model path
model.eval()

# Real-time callback
def denoise_callback(indata, outdata, frames, time, status):
    if status:
        print("⚠️", status)
    audio = torch.tensor(indata[:, 0]).unsqueeze(0)

    # Convert to Mel
    mel = mel_spec(audio)
    mel_db = to_db(mel).squeeze(0)

    # Denoise
    with torch.no_grad():
        denoised = model(mel_db.unsqueeze(0)).squeeze(0)

    # [Optional] Convert Mel back to waveform (placeholder – for real usage, use Griffin-Lim)
    # Here we use a dummy passthrough to output original (no reconstruction)
    outdata[:] = indata

# Start stream
print("🎧 Real-time noise suppression started. Speak now...")
with sd.Stream(channels=1, samplerate=SAMPLE_RATE, blocksize=NUM_SAMPLES, callback=denoise_callback):
    sd.sleep(10000)  # Run for 10 seconds