import torch
import torchaudio
from transformers import AudioSpectrogramTransformerModel, AutoProcessor

# Use a pretrained audio captioning model (e.g., AudioSpectrogramTransformer)
# This model is trained for audio captioning (e.g., on Clotho dataset)
model_id = "m-a-p/audio-captioning-base"
processor = AutoProcessor.from_pretrained(model_id)
model = AudioSpectrogramTransformerModel.from_pretrained(model_id)
model.eval()

# Load and resample audio (16 kHz mono)
waveform, sample_rate = torchaudio.load("data/environment.wav")
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)
waveform = waveform.mean(dim=0).unsqueeze(0)  # convert to mono

# Preprocess audio
inputs = processor(audios=waveform, sampling_rate=16000, return_tensors="pt")

# Generate caption
with torch.no_grad():
    output = model.generate(**inputs, max_length=20)

caption = processor.batch_decode(output, skip_special_tokens=True)[0]
print("🔊 Audio Caption:", caption)