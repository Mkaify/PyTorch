import cv2
import mediapipe as mp
import numpy as np
 
# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
 
# Angle calculation function
def calculate_angle(a, b, c):
    a = np.array(a)  # first point
    b = np.array(b)  # joint
    c = np.array(c)  # end point
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)
 
# Webcam loop
cap = cv2.VideoCapture(0)
print("🧘 Yoga pose correction running... Press 'q' to quit.")
 
while True:
    ret, frame = cap.read()
    if not ret:
        break
 
    # Convert to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
 
    if results.pose_landmarks:
        # Draw landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
 
        # Extract landmarks
        landmarks = results.pose_landmarks.landmark
        h, w, _ = image.shape
 
        # Get coordinates for left arm (shoulder, elbow, wrist)
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w,
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * h]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * w,
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * h]
 
        # Calculate angle at elbow
        angle = calculate_angle(shoulder, elbow, wrist)
 
        # Display angle and feedback
        cv2.putText(image, f"Elbow Angle: {int(angle)}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
 
        # Feedback based on angle range
        if 160 < angle < 180:
            feedback = "Good arm extension"
        else:
            feedback = "Straighten your arm"
 
        cv2.putText(image, feedback, (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
 
    cv2.imshow("Yoga Pose Correction", image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
 
cap.release()
cv2.destroyAllWindows()