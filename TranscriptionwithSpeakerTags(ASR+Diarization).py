from pyannote.audio import Pipeline
import whisper
import torch
import os

# Load diarization pipeline (requires Hugging Face token)
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization@2.1",
    use_auth_token="your_huggingface_token_here"  # Replace with your token
)

# Load ASR model (Whisper)
asr_model = whisper.load_model("base")

# Input file
audio_path = "data/conversation.wav"

# Step 1: Diarization
diarization = diarization_pipeline(audio_path)

# Step 2: Speaker segmentation list
segments = []
for segment, _, speaker in diarization.itertracks(yield_label=True):
    segments.append({
        "start": segment.start,
        "end": segment.end,
        "speaker": speaker
    })

# Step 3: Transcribe each segment
transcript = []
for seg in segments:
    # Crop audio per speaker turn
    output_file = "temp_segment.wav"
    command = f"ffmpeg -y -i {audio_path} -ss {seg['start']} -to {seg['end']} -ar 16000 -ac 1 {output_file} -loglevel quiet"
    os.system(command)

    result = asr_model.transcribe(output_file)
    seg["text"] = result["text"]
    transcript.append(seg)

# Step 4: Display speaker-attributed transcript
print("\n🗒️ Speaker-attributed Transcript:\n")
for seg in transcript:
    start = round(seg["start"], 1)
    end = round(seg["end"], 1)
    print(f"[{start}-{end}s] {seg['speaker']}: {seg['text']}")