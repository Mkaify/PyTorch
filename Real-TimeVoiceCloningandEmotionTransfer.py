import torch
import torchaudio
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import soundfile as sf
 
# Load components
tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
tts_model.eval()
 
# Load reference audio for speaker identity
ref_audio, sr = torchaudio.load("data/target_voice.wav")
ref_audio = ref_audio[:, :32000]  # ~2 seconds
speaker_inputs = processor(audio=ref_audio, sampling_rate=sr, return_tensors="pt")
speaker_embedding = tts_model.get_speaker_embeddings(**speaker_inputs)
 
# Define emotion prompt (this is synthetic, real emotion embeddings require dedicated models)
emotion_prompt = "happy"  # You could also try: "sad", "angry", "neutral"
text = f"[{emotion_prompt}] Hello! I am speaking in your voice, with a touch of {emotion_prompt}."
 
# Encode text
text_inputs = processor(text=text, return_tensors="pt")
 
# Generate speech conditioned on speaker
with torch.no_grad():
    generated_speech = tts_model.generate_speech(text_inputs["input_ids"], speaker_embeddings=speaker_embedding, vocoder=vocoder)
 
# Save audio
sf.write("outputs/voice_clone_emotion.wav", generated_speech.cpu().numpy(), 16000)
print(f"🧬 Cloned voice with '{emotion_prompt}' emotion saved to outputs/voice_clone_emotion.wav")