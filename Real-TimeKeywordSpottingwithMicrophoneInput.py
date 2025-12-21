import torch
import torch.nn as nn
import torchaudio.transforms as T
import sounddevice as sd
import numpy as np
 
# Keyword list and label map
KEYWORDS = ["yes", "no", "stop"]
LABELS = {0: "yes", 1: "no", 2: "stop"}
 
# Audio parameters
SAMPLE_RATE = 16000
DURATION = 1  # seconds
NUM_SAMPLES = SAMPLE_RATE * DURATION
 
# Load trained model or dummy
class KeywordCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(32, len(KEYWORDS))
 
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.conv(x)
        return self.fc(x.squeeze(-1).squeeze(-1))
 
# Instantiate or load your trained model
model = KeywordCNN()
model.eval()
 
# Mel-spectrogram transform (same as training)
mel_transform = T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=64)
 
# Real-time audio classification callback
def audio_callback(indata, frames, time, status):
    if status:
        print("⚠️", status)
    audio = torch.tensor(indata[:, 0]).float().unsqueeze(0)
    if audio.shape[1] < NUM_SAMPLES:
        audio = torch.nn.functional.pad(audio, (0, NUM_SAMPLES - audio.shape[1]))
 
    mel = mel_transform(audio).squeeze(0)
    with torch.no_grad():
        output = model(mel.unsqueeze(0))
        pred = torch.argmax(output, dim=1).item()
        print("🔊 Detected keyword:", LABELS[pred])
 
# Start microphone stream
print("🎙️ Listening for keywords... Speak 'yes', 'no', or 'stop'")
with sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=NUM_SAMPLES):
    sd.sleep(10000)  # Listen for 10 seconds