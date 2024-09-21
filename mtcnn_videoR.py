import cv2
from mtcnn.mtcnn import MTCNN

# Initialize the MTCNN face detector
detector = MTCNN()

# Capture video from the default camera (index 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream from camera.")
    exit()

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

        # Optionally, draw rectangles around key facial landmarks
        for key, point in face['keypoints'].items():
            # Draw a small rectangle around each landmark
            x, y = point
            rect_size = 10  # Size of the rectangle
            cv2.rectangle(frame, (x - rect_size // 2, y - rect_size // 2),
                          (x + rect_size // 2, y + rect_size // 2), (0, 0, 255), 2)

    # Show the frame with detected faces and landmarks
    cv2.imshow('Face Detection', frame)

    # Exit if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
