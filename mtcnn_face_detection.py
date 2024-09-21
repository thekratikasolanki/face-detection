import cv2
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt

# Load the image using OpenCV
# image = cv2.imread('C:/Users/HP/Downloads/archive/lfw-deepfunneled/lfw-deepfunneled/Zurab_Tsereteli/Zurab_Tsereteli_0001.jpg')

image = cv2.imread(r'C:\Users\HP\Downloads\WhatsApp Image 2024-09-16 at 8.13.54 AM.jpeg')



# Convert the image to RGB (MTCNN expects RGB images)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Initialize MTCNN detector
detector = MTCNN()

# Detect faces in the image
faces = detector.detect_faces(image_rgb)

# Draw bounding boxes around detected faces
for face in faces:
    x, y, width, height = face['box']
    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

# Display the image with bounding boxes
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Save the output image if necessary
cv2.imwrite('output.jpg', image)
