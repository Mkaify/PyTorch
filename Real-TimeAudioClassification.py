import torch
import torchaudio
import torchaudio.transforms as T
import sounddevice as sd
import numpy as np
 
# Parameters
SAMPLE_RATE = 16000
DURATION = 1  # seconds
NUM_CLASSES = 3  # Example: speech, clap, noise
 
# Dummy model (replace with trained model)
class DummyAudioClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 64, 32),
            nn.ReLU(),
            nn.Linear(32, NUM_CLASSES)
        )
 
    def forward(self, x):
        return self.net(x)
 
# Load your trained model here
model = DummyAudioClassifier()
model.eval()
 
# Define feature extractor
transform = nn.Sequential(
    T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=64),
    T.AmplitudeToDB()
)
 
# Real-time inference function
def classify_audio(audio):
    waveform = torch.tensor(audio).float().unsqueeze(0)
    mel = transform(waveform)
    with torch.no_grad():
        output = model(mel)
        pred = torch.argmax(output, dim=1).item()
    return pred
 
# Real-time stream
def callback(indata, frames, time, status):
    if status:
        print(status)
    indata = indata[:, 0]
    pred = classify_audio(indata)
    labels = ["speech", "clap", "noise"]
    print("Prediction:", labels[pred])
 
print("🎙️ Listening for 10 seconds... (make a sound)")
with sd.InputStream(callback=callback, channels=1, samplerate=SAMPLE_RATE, blocksize=SAMPLE_RATE * DURATION):
    sd.sleep(10000)  # 10 seconds