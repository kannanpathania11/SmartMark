import cv2
import face_recognition

print("face_recognition_models imported successfully")

# Load first image
img = cv2.imread("/Users/chitvan/Downloads/source code/chitvan1.jpg")
if img is None:
    print("Error: Could not read image 'Messi1.webp'")

# Convert to RGB
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_encoding = face_recognition.face_encodings(rgb_img)[0]

# Load second image
img2 = cv2.imread("/Users/chitvan/Downloads/source code/images/Chitvan Bhardwaj.jpg")  # Use absolute path if needed
if img2 is None:
    print("Error: Could not read image 'images/Messi.webp'")

# Convert to RGB
rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img_encoding2 = face_recognition.face_encodings(rgb_img2)[0]

# Compare faces
result = face_recognition.compare_faces([img_encoding], img_encoding2)
print("Result: ", result)

# Display images
cv2.imshow("Img", img)
cv2.imshow("Img 2", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()