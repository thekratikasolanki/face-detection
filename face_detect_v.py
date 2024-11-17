import cv2
from mtcnn.mtcnn import MTCNN
from playsound import playsound
import os

# Initialize the MTCNN face detector
detector = MTCNN()

# Function to play sound when face is detected
def play_sound():
    sound_path = 'alert_sound.mpeg'  # Replace with the path to your sound file
    playsound(sound_path)

# Capture video from the default camera (index 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream from camera.")
    exit()

# Directory to save the detected face images
output_dir = 'detected_faces'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

face_counter = 0  # Counter for saved face images

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame from camera.")
        break

    # Detect faces in the frame
    faces = detector.detect_faces(frame)

    # Loop through each face and draw a bounding box around it
    for face in faces:
        x, y, width, height = face['box']
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # Extract the detected face from the frame
        face_image = frame[y:y + height, x:x + width]

        # Save the detected face as an image
        face_image_path = os.path.join(output_dir, f"face_{face_counter}.jpg")
        cv2.imwrite(face_image_path, face_image)
        print(f"Face {face_counter} saved to {face_image_path}")

        # Play sound when face is detected
        play_sound()

        face_counter += 1

    # Show the frame with detected faces
    cv2.imshow('Face Detection', frame)

    # Exit if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()