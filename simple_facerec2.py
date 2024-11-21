import face_recognition
import cv2
import os
import glob
import numpy as np


class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

        # Resize frame for faster speed
        self.frame_resizing = 0.25

    def load_encoding_images(self, images_path):
        """
        Load encoding images from a specified path.
        :param images_path: Path to the directory containing the images.
        """
        # Load all image file paths
        images_path = glob.glob(os.path.join(images_path, "*.*"))
        print(f"{len(images_path)} encoding images found.")

        # Iterate through each image to extract and store face encodings
        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Extract the file name (without extension) to use as the name
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)

            # Generate face encodings
            face_encodings = face_recognition.face_encodings(rgb_img)
            if face_encodings:  # Ensure at least one face encoding is found
                img_encoding = face_encodings[0]

                # Append encoding and name
                self.known_face_encodings.append(img_encoding)
                self.known_face_names.append(filename)
                print(f"Encoding for {filename} loaded successfully.")
            else:
                print(f"No face found in {filename}, skipping this image.")

        print("All encoding images loaded.")

    def detect_known_faces(self, frame):
        """
        Detect faces in the frame and match them against known faces.
        :param frame: A frame from a video or an image to process.
        :return: List of face locations and corresponding names.
        """
        # Resize frame for faster face detection
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)

        # Convert frame to RGB
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect face locations and encodings in the frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Compare detected face encodings with known encodings
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            # Use the closest matching face based on distance
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]

            face_names.append(name)

        # Scale back the face locations to the original frame size
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names