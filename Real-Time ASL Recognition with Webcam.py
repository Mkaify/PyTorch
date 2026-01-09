import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
 
# Labels A–Z
labels = [chr(i) for i in range(65, 91)]  # A to Z
 
# Load trained ASL model
class SignLanguageCNN(nn.Module):
    def __init__(self, num_classes=26):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
 
    def forward(self, x):
        return self.net(x)
 
model = SignLanguageCNN()
model.load_state_dict(torch.load("asl_model.pth", map_location="cpu"))  # Replace with your trained model path
model.eval()
 
# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
 
# Webcam setup
cap = cv2.VideoCapture(0)
print("🖐️ ASL recognition started... Show a letter in the ROI. Press 'q' to quit.")
 
while True:
    ret, frame = cap.read()
    if not ret:
        break
 
    # Define Region of Interest (ROI)
    x1, y1, x2, y2 = 100, 100, 300, 300
    roi = frame[y1:y2, x1:x2]
    roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
 
    # Preprocess and predict
    input_tensor = transform(roi_pil).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
        label = labels[pred]
 
    # Display prediction and bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
    cv2.putText(frame, f"Prediction: {label}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
 
    cv2.imshow("Real-Time ASL Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()