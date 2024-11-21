import cv2
import streamlit as st
import face_recognition as frg
from simple_facerec import SimpleFacerec
import yaml
import numpy as np
import os
import pickle

# Function to save user data for sign up
def save_user_data(name, enrollment, user_class, semester, image_path):
    if not os.path.exists("users"):
        os.makedirs("users")

    user_data = {
        "name": name,
        "enrollment": enrollment,
        "class": user_class,
        "semester": semester,
        "image_path": image_path
    }

    # Save user data in a pickle file for persistence
    with open(f"users/{enrollment}.pkl", "wb") as f:
        pickle.dump(user_data, f)

# Function to load user data
def load_user_data(enrollment):
    try:
        with open(f"users/{enrollment}.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

# Load configuration from YAML file
cfg = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
PICTURE_PROMPT = cfg['INFO']['PICTURE_PROMPT']
WEBCAM_PROMPT = cfg['INFO']['WEBCAM_PROMPT']

# Initialize SimpleFacerec for face recognition
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")  # Set your image path here

# Set Streamlit page config
st.set_page_config(layout="wide")

# Streamlit interface
st.title("SmartMark - Real-Time Face Recognition App")
st.subheader("This app detects known faces in real-time using your webcam or uploaded pictures.")

# Menu for navigating the app
menu = ["Login", "Signup", "Dashboard"]
choice = st.sidebar.selectbox("Select an option", menu)

# Use session state to remember the login state
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# Store recognized names to avoid repeating attendance
if 'recognized_names' not in st.session_state:
    st.session_state['recognized_names'] = set()

# List to keep track of names whose attendance has been marked
if 'marked_attendance_names' not in st.session_state:
    st.session_state['marked_attendance_names'] = []

def log_attendance(name, enrollment):
    st.write(f"Attendance registered for {name} (ID: {enrollment})")
    if name not in st.session_state['marked_attendance_names']:
        st.session_state['marked_attendance_names'].append(name)

# Signup Section
if choice == "Signup":
    st.subheader("Sign Up")
    
    # User inputs
    name = st.text_input("Name")
    enrollment = st.text_input("Enrollment Number (ID)")
    user_class = st.text_input("Class")
    semester = st.text_input("Semester")
    
    # Option to upload image or use webcam to capture image
    signup_option = st.radio("Choose Image Source", ["Upload Image", "Capture Image via Webcam"])

    if signup_option == "Upload Image":
        user_image = st.file_uploader("Upload your image", type=['jpg', 'png', 'jpeg'])
    else:
        # Webcam capture for signup
        st.write("Click 'Capture' to take a photo with your webcam.")
        capture_button = st.button("Capture Image")
        
        if capture_button:
            # Open the webcam and capture an image
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            if ret:
                st.image(frame, channels="BGR")  # Display captured frame
                # Save image after capture
                image_path = f"images/{enrollment}.jpg"
                cv2.imwrite(image_path, frame)
                cap.release()
                st.success(f"Captured image saved at {image_path}")
            else:
                st.error("Failed to capture image. Try again.")
        
    # Signup button
    if st.button("Sign Up") and (user_image or capture_button):
        if name and enrollment and user_class and semester:
            # If user uploaded image or captured via webcam
            if signup_option == "Upload Image" and user_image:
                img = frg.load_image_file(user_image)
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                image_path = f"images/{enrollment}.jpg"
                cv2.imwrite(image_path, rgb_img)
            elif signup_option == "Capture Image via Webcam":
                img = cv2.imread(image_path)
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Save user data with the image path
            save_user_data(name, enrollment, user_class, semester, image_path)
            st.success(f"User {name} signed up successfully! Image saved at {image_path}")
        else:
            st.error("Please fill all the fields.")

# Login Section
elif choice == "Login":
    st.subheader("Login")
    
    # Login inputs
    enrollment_number = st.text_input("Enter your Enrollment Number (ID)")
    
    # Login button
    if st.button("Login"):
        user_data = load_user_data(enrollment_number)
        if user_data:
            st.session_state['logged_in'] = True
            st.session_state['user_data'] = user_data
            st.success(f"Welcome {user_data['name']}!")
        else:
            st.error("Invalid Enrollment Number! Please try again.")

# Dashboard Section
elif choice == "Dashboard":
    if st.session_state['logged_in']:
        st.subheader("Dashboard")
        view_attendance = st.button("View Attendance")

        if view_attendance:
            st.subheader("Attendance List")
            if st.session_state['marked_attendance_names']:
                st.write("Attendance has been marked for the following students:")
                for name in st.session_state['marked_attendance_names']:
                    st.write(name)
            else:
                st.write("No attendance has been marked yet.")
    else:
        st.warning("Please login to access the dashboard.")

# Face Recognition Section
if st.session_state['logged_in']:
    st.subheader("Face Recognition")

    # Choose between Webcam or Picture input
    choice = st.radio("Choose Input Type", ["Webcam", "Picture"])
    TOLERANCE = st.slider("Tolerance", 0.0, 1.0, 0.5, 0.01)

    # Display student information
    st.sidebar.title("Student Information")
    st.sidebar.info(f"Name: {st.session_state['user_data']['name']}")
    st.sidebar.success(f"ID: {st.session_state['user_data']['enrollment']}")

    frame_placeholder = st.empty()

    if choice == "Picture":
        st.write(PICTURE_PROMPT)
        uploaded_images = st.file_uploader("Upload Picture(s)", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)

        if uploaded_images:
            for image in uploaded_images:
                img = frg.load_image_file(image)
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                face_locations, face_names = sfr.detect_known_faces(rgb_img)

                for face_loc, name in zip(face_locations, face_names):
                    y1, x2, y2, x1 = face_loc
                    cv2.putText(rgb_img, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
                    cv2.rectangle(rgb_img, (x1, y1), (x2, y2), (0, 200, 0), 2)
                    if name not in st.session_state['recognized_names']:
                        st.session_state['recognized_names'].add(name)
                        log_attendance(name, st.session_state['user_data']['enrollment'])

                st.image(rgb_img)

    elif choice == "Webcam":
        st.write(WEBCAM_PROMPT)
        device_id = st.sidebar.selectbox("Select Webcam Device ID", [0, 1, 2, 3], index=0)
        cap = cv2.VideoCapture(device_id)

        run = st.checkbox("Start Video Stream")
        while run:
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to capture frame.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations, face_names = sfr.detect_known_faces(rgb_frame)

            for face_loc, name in zip(face_locations, face_names):
                if name not in st.session_state['recognized_names']:
                    st.session_state['recognized_names'].add(name)
                    log_attendance(name, st.session_state['user_data']['enrollment'])

            st.image(rgb_frame)
        cap.release()
