import torch
import torchaudio
import sounddevice as sd
import numpy as np
from torchaudio.pipelines import VADIterator

# Load torchaudio’s VAD iterator (based on Silero VAD)
vad_model = torchaudio.pipelines.SILERO_VAD.get_model()

# Sample rate required
SAMPLE_RATE = 16000
FRAME_SIZE = 512  # ~32ms
CHUNK_DURATION = 1  # seconds

# Capture audio (simulate real-time)
print("🎙️ Speak for 5 seconds...")
audio = sd.rec(int(SAMPLE_RATE * CHUNK_DURATION * 5), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
sd.wait()
waveform = torch.tensor(audio.T)

# Slice into frames
frames = waveform.unfold(1, FRAME_SIZE, FRAME_SIZE)

# Apply VAD per frame
speech_frames = []
for frame in frames[0]:
    is_speech = vad_model(frame.unsqueeze(0), sample_rate=SAMPLE_RATE).item()
    speech_frames.append(is_speech)

# Print VAD timeline
print("\n🧠 VAD Output:")
for i, is_speech in enumerate(speech_frames):
    ts = round(i * FRAME_SIZE / SAMPLE_RATE, 2)
    label = "Speech" if is_speech else "Silence"
    print(f"{ts:5.2f}s - {label}")