import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio
 
# Load pretrained model and processor
model_name = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)
 
# Load audio file (must be 16kHz mono)
waveform, sample_rate = torchaudio.load("data/sample_speech.wav")
 
# Resample if not 16kHz
if sample_rate != 16000:
    waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
 
# Prepare input
input_values = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").input_values
 
# Inference
with torch.no_grad():
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
 
# Decode predicted token IDs to text
transcription = processor.decode(predicted_ids[0])
print("Transcription:", transcription)