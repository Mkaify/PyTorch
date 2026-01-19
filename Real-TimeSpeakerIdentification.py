import torch
import torchaudio
import sounddevice as sd
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# Load pretrained Wav2Vec2 model as speaker embedder
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
model.eval()

# Audio settings
SAMPLE_RATE = 16000
DURATION = 2  # seconds
NUM_SAMPLES = SAMPLE_RATE * DURATION

# Speaker embedding function
def get_embedding(audio):
    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

# Enrollment database (simulate 2 speakers)
print("🎙️ Record 2 enrolled speakers (2 seconds each)...")

enrolled_embeddings = {}
for speaker in ["Alice", "Bob"]:
    print(f"Recording for {speaker}...")
    audio = sd.rec(NUM_SAMPLES, samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    waveform = torch.tensor(audio[:, 0])
    embedding = get_embedding(waveform)
    enrolled_embeddings[speaker] = embedding

# Real-time speaker identification
print("\n🎧 Listening for speaker...")

def identify_speaker(audio):
    test_embedding = get_embedding(torch.tensor(audio[:, 0]))
    scores = {}
    for name, ref_emb in enrolled_embeddings.items():
        sim = torch.cosine_similarity(ref_emb, test_embedding, dim=0).item()
        scores[name] = sim
    best_match = max(scores, key=scores.get)
    return best_match, scores[best_match]

# Real-time stream (one-time capture for simplicity)
print("Say something...")
audio = sd.rec(NUM_SAMPLES, samplerate=SAMPLE_RATE, channels=1, dtype='float32')
sd.wait()

speaker, confidence = identify_speaker(audio)
print(f"\n🔍 Identified Speaker: {speaker} (confidence: {confidence:.2f})")