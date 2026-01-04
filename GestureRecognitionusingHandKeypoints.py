import cv2
import mediapipe as mp
import numpy as np
 
# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
 
# Simple rule-based gesture classifier
def classify_gesture(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    wrist = landmarks[0]
 
    # Convert to numpy
    tips = np.array([thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip])
    distances = np.linalg.norm(tips - wrist, axis=1)
 
    if all(dist < 0.1 for dist in distances):
        return "Fist"
    elif distances[0] > 0.2 and all(dist < 0.15 for dist in distances[1:]):
        return "Thumbs Up"
    elif all(dist > 0.2 for dist in distances):
        return "Open Palm"
    else:
        return "Unknown"
 
# Start webcam
cap = cv2.VideoCapture(0)
print("🖐️ Gesture recognition running... Press 'q' to quit.")
 
while True:
    ret, frame = cap.read()
    if not ret:
        break
 
    # Convert image to RGB and process
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
 
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
 
            # Extract normalized landmark positions
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            gesture = classify_gesture(landmarks)
 
            # Display gesture label
            cv2.putText(frame, gesture, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
 
    cv2.imshow("Hand Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
 
cap.release()
cv2.destroyAllWindows()