import cv2
import mediapipe as mp
import numpy as np
 
# Initialize MediaPipe Face Mesh
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils
 
# Indices for eye landmarks (right eye only for simplicity)
RIGHT_EYE = [33, 160, 158, 133, 153, 144]  # Eye contour
RIGHT_IRIS = [469, 470, 471, 472]          # Iris
 
# EAR (eye aspect ratio) threshold for blink detection
def calculate_ear(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)
 
# Gaze estimation (compare iris x-position to eyelid x-range)
def estimate_gaze(eye, iris):
    eye_x = [eye[0][0], eye[3][0]]
    iris_x = iris[:, 0].mean()
    if iris_x < np.mean(eye_x) - 0.01:
        return "Right"
    elif iris_x > np.mean(eye_x) + 0.01:
        return "Left"
    else:
        return "Center"
 
# Webcam stream
cap = cv2.VideoCapture(0)
print("👀 Eye tracking started. Press 'q' to quit.")
 
while True:
    ret, frame = cap.read()
    if not ret:
        break
 
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
 
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Convert landmarks to NumPy
            h, w, _ = frame.shape
            landmarks = np.array([[lm.x * w, lm.y * h] for lm in face_landmarks.landmark])
 
            eye = landmarks[RIGHT_EYE]
            iris = landmarks[RIGHT_IRIS]
 
            ear = calculate_ear(eye)
            blink = "Blink" if ear < 0.2 else "Open"
 
            gaze = estimate_gaze(eye, iris)
 
            # Draw eye contour
            for p in eye:
                cv2.circle(frame, tuple(p.astype(int)), 2, (255, 255, 255), -1)
 
            # Draw iris center
            iris_center = iris.mean(axis=0).astype(int)
            cv2.circle(frame, tuple(iris_center), 3, (0, 255, 0), -1)
 
            # Display text
            cv2.putText(frame, f"{gaze} | {blink}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
 
    cv2.imshow("Eye Gaze and Blink Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
 
cap.release()
cv2.destroyAllWindows()