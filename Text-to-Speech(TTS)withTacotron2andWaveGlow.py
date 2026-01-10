import torch
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
 
# Load Tacotron2 and WaveGlow from NVIDIA's pretrained models
tacotron2 = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_tacotron2')
waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')
tacotron2.eval()
waveglow.eval()
 
# Remove weight normalization for faster inference
waveglow.remove_weightnorm()
 
# Load text processor
from tacotron2.text import text_to_sequence
 
# Text to synthesize
text = "Hello! Welcome to the PyTorch text-to-speech demo."
 
# Convert text to sequence
sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
sequence = torch.from_numpy(sequence).long()
 
# Generate mel spectrogram with Tacotron2
with torch.no_grad():
    mel_outputs, mel_outputs_postnet, _, alignments = tacotron2.infer(sequence)
 
# Generate audio with WaveGlow
with torch.no_grad():
    audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
 
# Convert to CPU and numpy
audio = audio[0].data.cpu().numpy()
audio = audio / np.abs(audio).max()  # normalize
 
# Save or play audio
import soundfile as sf
sf.write("tts_output.wav", audio, 22050)
 
# Optional: play in notebook
ipd.display(ipd.Audio(audio, rate=22050))