import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForSeq2SeqLM
 
# Step 1: Load a pretrained sound tagging model
tagger = torch.hub.load('qiuqiangkong/audioset_tagging_cnn', 'resnet38')
tagger.eval()
 
# Audio preprocessing
waveform, sr = torchaudio.load("data/city_scene.wav")
waveform = torchaudio.functional.resample(waveform, sr, 32000)
waveform = waveform[:, :32000]  # 1 second
mono_waveform = waveform.mean(dim=0).unsqueeze(0)
 
# Tag the audio (outputs AudioSet tags)
with torch.no_grad():
    tags = tagger(mono_waveform)
topk = torch.topk(tags, k=5).indices[0]
event_labels = [tagger.labels[i] for i in topk]
 
# Step 2: Construct prompt from tags
tag_phrase = ", ".join(event_labels)
prompt = f"Describe a scene where you hear: {tag_phrase}."
 
# Step 3: Use a text-to-text model for narration
processor = AutoProcessor.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
 
inputs = processor(prompt, return_tensors="pt")
output = model.generate(**inputs, max_length=50)
narration = processor.batch_decode(output, skip_special_tokens=True)[0]
 
# Output
print("🔊 Sound Tags:", event_labels)
print("\n📖 Scene Narration:", narration)