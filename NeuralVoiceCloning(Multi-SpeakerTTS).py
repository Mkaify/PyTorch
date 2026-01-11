import torch
import torchaudio
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import numpy as np
import soundfile as sf

# Load pretrained model and vocoder
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Load speaker embedding (precomputed or synthetic)
# Option 1: Load from a file of 1.6s speech (use Microsoft's speaker encoder)
# Option 2: Use provided embeddings
speaker_embeddings = torch.randn(1, 512)  # ← Use real embedding in real projects

# Prepare input text
text = "Hello there, I am your personal assistant with a cloned voice."
inputs = processor(text=text, return_tensors="pt")

# Generate mel spectrogram conditioned on speaker embedding
with torch.no_grad():
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

# Save audio
sf.write("cloned_voice.wav", speech.numpy(), 16000)

# Optional: play in notebook
import IPython.display as ipd
ipd.display(ipd.Audio(speech.numpy(), rate=16000))