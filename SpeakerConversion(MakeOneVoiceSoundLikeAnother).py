import torch
import torchaudio
from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech, SpeechT5HifiGan
import soundfile as sf
 
# Load speech-to-speech model (AutoVC-style) and vocoder
model = SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_s2s")
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_s2s")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
model.eval()
 
# Load reference speaker embedding (e.g., 2s audio from target speaker)
ref_waveform, sr = torchaudio.load("data/speaker_B_reference.wav")
ref_waveform = ref_waveform[:, :32000]
speaker_input = processor(audio=ref_waveform, sampling_rate=sr, return_tensors="pt")
speaker_embedding = model.get_speaker_embeddings(**speaker_input)
 
# Load source voice (the voice to convert)
src_waveform, _ = torchaudio.load("data/source_speech.wav")
src_waveform = src_waveform[:, :32000]  # 2 seconds max
inputs = processor(audio=src_waveform, sampling_rate=16000, return_tensors="pt")
 
# Perform speaker conversion
with torch.no_grad():
    converted = model.generate_speech(inputs["input_values"], speaker_embeddings=speaker_embedding, vocoder=vocoder)
 
# Save output
sf.write("outputs/converted_voice.wav", converted.cpu().numpy(), 16000)
print("🎤 Voice converted and saved to outputs/converted_voice.wav")