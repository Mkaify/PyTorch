import torchaudio
import torchaudio.pipelines
import soundfile as sf
import torch

# Load Demucs pretrained model (v4 is more accurate)
bundle = torchaudio.pipelines.DEMUCS_HTDEMUX
model = bundle.get_model()
model.eval()

# Load mixed audio
audio_path = "data/song_clip.wav"  # stereo WAV, 44.1kHz preferred
waveform, sample_rate = torchaudio.load(audio_path)

# Ensure stereo
if waveform.shape[0] == 1:
    waveform = waveform.repeat(2, 1)

# Perform separation
with torch.no_grad():
    sources = model(waveform.unsqueeze(0))  # shape: [1, 4, 2, T]

# Save separated stems
stem_names = ["drums", "bass", "other", "vocals"]
output_dir = "./outputs/separated_stems"
os.makedirs(output_dir, exist_ok=True)

for i, name in enumerate(stem_names):
    stem = sources[0, i]  # [2, T]
    sf.write(f"{output_dir}/{name}.wav", stem.T.cpu().numpy(), samplerate=sample_rate)

print("🎼 Separated sources saved to:", output_dir)