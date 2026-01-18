# Required installation (in terminal):
# pip install pyannote.audio
# Hugging Face authentication required for some models

from pyannote.audio import Pipeline

# Load pretrained diarization pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                    use_auth_token="your_huggingface_token_here")  # Replace with your HF token

# Input audio file (mono, 16 kHz WAV)
audio_file = "data/meeting_sample.wav"

# Run diarization
diarization = pipeline(audio_file)

# Print speaker segments
print("🔊 Speaker segments:")
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{speaker}: {turn.start:.1f}s - {turn.end:.1f}s")

# Optional: visualize using matplotlib
from pyannote.core import notebook
notebook.crop = diarization.get_timeline().extent()
diarization.plot()