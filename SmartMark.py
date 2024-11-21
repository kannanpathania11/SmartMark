import cv2
import streamlit as st
import face_recognition as frg
from simple_facerec import SimpleFacerec
import yaml
import numpy as np
import os
import pickle
import json
import pandas as pd
import requests

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

# Function to fetch all user data and cache it in session state
def fetch_all_users():
    if 'users_data' not in st.session_state:
        st.session_state['users_data'] = []
        if os.path.exists("users"):
            for file in os.listdir("users"):
                if file.endswith(".pkl"):
                    with open(f"users/{file}", "rb") as f:
                        st.session_state['users_data'].append(pickle.load(f))
    return st.session_state['users_data']

# Function to initialize SimpleFacerec and cache it in session state
def initialize_face_recognition():
    if 'sfr_initialized' not in st.session_state:
        st.session_state['sfr'] = SimpleFacerec()
        st.session_state['sfr'].load_encoding_images("images/")
        st.session_state['sfr_initialized'] = True

# Load timetable from JSON file
@st.cache_data
def load_timetable():
    with open("timetable.json", "r") as file:
        return pd.json_normalize(data=json.load(file)["timetable"], record_path="slots", meta="day")

# Initialize SimpleFacerec
initialize_face_recognition()

# Set Streamlit page config
st.set_page_config(layout="wide")

# Streamlit interface
st.title("SmartMark - Real-Time Face Recognition App")
st.subheader("This app detects known faces in real-time using your webcam or uploaded pictures.")

# Menu for navigating the app
menu = ["Login", "Signup", "Dashboard", "Users", "Face Recognition"]
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

# Function to log attendance
def log_attendance(name, enrollment):
    st.write(f"Attendance registered for {name} (ID: {enrollment})")
    if name not in st.session_state['marked_attendance_names']:
        st.session_state['marked_attendance_names'].append(name)

# Users Section
if choice == "Users":
    st.subheader("Users")
    users_data = fetch_all_users()

    if users_data:
        for user in users_data:
            col1, col2 = st.columns([1, 3])

            # Display user image
            with col1:
                if os.path.exists(user["image_path"]):
                    st.image(user["image_path"], width=100, caption=user["name"])
                else:
                    st.text("Image not found")

            # Display user details
            with col2:
                st.text(f"Name: {user['name']}")
                st.text(f"Enrollment: {user['enrollment']}")
                st.text(f"Class: {user['class']}")
                st.text(f"Semester: {user['semester']}")
    else:
        st.warning("No users found. Please sign up to add users.")

# Signup Section
elif choice == "Signup":
    st.subheader("Sign Up")
    
    # User inputs
    name = st.text_input("Name")
    enrollment = st.text_input("Enrollment Number (ID)")
    user_class = st.text_input("Class")
    semester = st.text_input("Semester")
    user_image = st.file_uploader("Upload your image", type=['jpg', 'png', 'jpeg'])

    # Signup button
    if st.button("Sign Up"):
        if name and enrollment and user_class and semester and user_image:
            # Load the uploaded image
            img = frg.load_image_file(user_image)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Create the 'images' directory if it doesn't exist
            if not os.path.exists("images"):
                os.makedirs("images")

            # Define image path using the enrollment number
            image_path = f"images/{enrollment}.jpg"
            
            # Save the image to the 'images' directory
            cv2.imwrite(image_path, rgb_img)

            # Save user data with the image path
            save_user_data(name, enrollment, user_class, semester, image_path)

            # Refresh user data cache
            if 'users_data' in st.session_state:
                st.session_state['users_data'].append({
                    "name": name,
                    "enrollment": enrollment,
                    "class": user_class,
                    "semester": semester,
                    "image_path": image_path
                })
            else:
                fetch_all_users()

            # Reload face encodings
            if 'sfr' in st.session_state:
                st.session_state['sfr'].load_encoding_images("images/")
            
            st.success(f"User {name} signed up successfully! Image saved at {image_path}")
        else:
            st.error("Please fill all the fields and upload your image.")

# Login Section
elif choice == "Login":
    st.subheader("Login")
    
    # Login inputs
    enrollment_number = st.text_input("Enter your Enrollment Number (ID)")
    
    # Login button
    if st.button("Login"):
        users_data = fetch_all_users()
        user_data = next((user for user in users_data if user['enrollment'] == enrollment_number), None)
        if user_data:
            st.session_state['logged_in'] = True
            st.session_state['user_data'] = user_data
            st.session_state['enrollment_number'] = user_data['enrollment']  # Store enrollment number
            st.success(f"Welcome {user_data['name']}!")
        else:
            st.error("Invalid enrollment number. Please try again.")

# Facial Recognition Section
elif choice == "Face Recognition":
    st.subheader("Face Recognition - Mark Attendance")
    
    # Ensure enrollment number is available
    enrollment_number = st.session_state.get('enrollment_number', None)
    if enrollment_number is None:
        st.error("Please log in first to mark attendance.")
    else:
        # Initialize webcam for face recognition
        video_capture = cv2.VideoCapture(0)

        while True:
            ret, frame = video_capture.read()
            if not ret:
                st.write("Error: Failed to access the webcam.")
                break

            # Detect faces in the frame
            face_locations, face_names = st.session_state['sfr'].detect_known_faces(frame)

            # Display the resulting frame
            st.image(frame, channels="BGR", use_container_width=True)

            # If faces are recognized, log attendance
            if face_names:
                for name in face_names:
                    log_attendance(name, enrollment_number)
                st.write("Attendance marked successfully.")
                break

        video_capture.release()
        cv2.destroyAllWindows()

# Dashboard Section
elif choice == "Dashboard":
    st.subheader("Student Dashboard")
   
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


    # Load the timetable
    timetable = load_timetable()
    
    # Display timetable by day
    days = timetable["day"].unique()
    selected_day = st.selectbox("Select Day", days)

    filtered_timetable = timetable[timetable["day"] == selected_day]

    st.subheader(f"Timetable for {selected_day}")

    # Create spacious grid layout for the timetable
    for index, row in filtered_timetable.iterrows():
        time = row["time"]
        subject = row["subject"]
        teacher = row.get("teacher", "")  # Remove "nan" for missing teacher field

        # Assign colors dynamically based on subject or time
        if subject == "Break":
            bg_color = "#FFD700"  # Gold for breaks
        else:
            bg_color = "#ADD8E6"  # Light blue for classes

        # Display timetable in a card-like style with more spacing
        with st.container():
            cols = st.columns([1, 3, 2, 2])  # Adjusted grid: Time, Subject, Teacher, Button
            with cols[0]:
                st.markdown(f"<div style='background-color:{bg_color}; padding:15px; border-radius:8px; text-align:center; font-weight:bold;'>{time}</div>", unsafe_allow_html=True)
            with cols[1]:
                st.markdown(f"<div style='background-color:{bg_color}; padding:15px; border-radius:8px; text-align:center; font-weight:bold;'>{subject}</div>", unsafe_allow_html=True)
            with cols[2]:
                st.markdown(f"<div style='background-color:{bg_color}; padding:15px; border-radius:8px; text-align:center; font-weight:bold;'>{teacher}</div>", unsafe_allow_html=True)

            with cols[3]:  # Mark Attendance Button
                if st.button(f"Mark Attendance for {subject}", key=f"attend-{index}"):
                    st.session_state['subject'] = subject  # Store the subject in session state
                    st.session_state['attendance_marked'] = False
                    st.rerun()  # Trigger page reload to go to facial recognition

# Facial Recognition Section
elif choice == "Face Recognition":
    st.subheader("Face Recognition - Mark Attendance")

    # Ensure enrollment number is available
    enrollment_number = st.session_state.get('enrollment_number', None)
    if enrollment_number is None:
        st.error("Please log in first to mark attendance.")
    else:
        # Initialize webcam for face recognition
        video_capture = cv2.VideoCapture(0)

        while True:
            ret, frame = video_capture.read()
            if not ret:
                st.write("Error: Failed to access the webcam.")
                break

            # Use SimpleFacerec to detect faces and recognize known ones
            face_locations, face_names = st.session_state['sfr'].detect_known_faces(frame)

            # Display the resulting frame with Streamlit
            st.image(frame, channels="BGR", use_container_width=True)

            # If faces are recognized, log attendance
            if face_names:
                for name in face_names:
                    if name != "Unknown" and name not in st.session_state['marked_attendance_names']:
                        log_attendance(name, enrollment_number)
                        st.session_state['marked_attendance_names'].append(name)
                st.write("Attendance marked successfully.")
                break

        video_capture.release()
        cv2.destroyAllWindows()