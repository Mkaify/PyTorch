import torch
import torchaudio
import torchvision.transforms as transforms
import cv2
from transformers import AutoProcessor, AutoModelForImageClassification
from transformers import Wav2Vec2Processor, HubertForSequenceClassification
 
# Load audio-based emotion model (e.g., HuBERT SER)
audio_model = HubertForSequenceClassification.from_pretrained("superb/hubert-large-superb-er")
audio_processor = Wav2Vec2Processor.from_pretrained("superb/hubert-large-superb-er")
audio_model.eval()
 
# Load image-based emotion model (e.g., ViT face emotion)
image_model = AutoModelForImageClassification.from_pretrained("nateraw/vit-face-expression")
image_processor = AutoProcessor.from_pretrained("nateraw/vit-face-expression")
image_model.eval()
 
# Load video and extract audio
video_path = "data/emotion_video.mp4"
audio_path = "temp.wav"
os.system(f"ffmpeg -y -i {video_path} -ar 16000 -ac 1 {audio_path} -loglevel quiet")
 
# Load and preprocess audio
waveform, sr = torchaudio.load(audio_path)
inputs_audio = audio_processor(waveform, sampling_rate=sr, return_tensors="pt")
with torch.no_grad():
    audio_logits = audio_model(**inputs_audio).logits
audio_emotion = torch.argmax(audio_logits, dim=1).item()
 
# Process video frame (extract one middle frame for demo)
cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
ret, frame = cap.read()
cap.release()
 
# Preprocess visual input
inputs_image = image_processor(images=frame, return_tensors="pt")["pixel_values"]
with torch.no_grad():
    image_logits = image_model(inputs_image).logits
image_emotion = torch.argmax(image_logits, dim=1).item()
 
# Combine predictions (simple fusion strategy)
# Map to common label space (manually aligned for this example)
label_map = ["angry", "happy", "neutral", "sad"]
 
final_emotion = label_map[max(audio_emotion, image_emotion)]  # simple fusion
print(f"\n🧠 Predicted Multimodal Emotion: {final_emotion.upper()}")