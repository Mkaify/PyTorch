import whisper
import torchaudio
import torch
import os
import soundfile as sf
from transformers import Wav2Vec2FeatureExtractor, HubertForSequenceClassification

# Load ASR model (Whisper)
asr_model = whisper.load_model("base")

# Load emotion classifier (HuBERT for emotion)
emotion_model = HubertForSequenceClassification.from_pretrained("superb/hubert-large-superb-er")
emotion_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-large-superb-er")
emotion_model.eval()

# Emotion label map
emotion_labels = ["angry", "happy", "neutral", "sad"]

# Function to predict emotion from waveform
def predict_emotion(waveform, sample_rate):
    inputs = emotion_extractor(waveform, sampling_rate=sample_rate, return_tensors="pt")
    with torch.no_grad():
        logits = emotion_model(**inputs).logits
    pred = torch.argmax(logits, dim=1).item()
    return emotion_labels[pred]

# Step 1: Transcribe + segment (Whisper gives timestamps)
audio_file = "data/emotional_speech.wav"
result = asr_model.transcribe(audio_file, word_timestamps=False)

# Step 2: For each segment, extract audio and predict emotion
segments = result["segments"]
final_output = []

for i, seg in enumerate(segments):
    start = seg["start"]
    end = seg["end"]
    text = seg["text"]

    # Extract audio segment using torchaudio
    waveform, sr = torchaudio.load(audio_file)
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    segment_waveform = waveform[:, start_sample:end_sample]

    # Predict emotion
    emotion = predict_emotion(segment_waveform, sr)

    final_output.append({
        "start": round(start, 1),
        "end": round(end, 1),
        "text": text.strip(),
        "emotion": emotion
    })

# Step 3: Print emotionally tagged transcript
print("\n🗒️ Emotion-Aware Transcript:\n")
for seg in final_output:
    print(f"[{seg['start']}-{seg['end']}s] ({seg['emotion'].upper()}): {seg['text']}")