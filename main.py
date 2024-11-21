import streamlit as st
import pandas as pd
import json
import requests
import cv2
from simple_facerec import SimpleFacerec  # Import your SimpleFacerec class

# Load timetable
@st.cache_data
def load_timetable():
    with open("timetable.json", "r") as file:
        return pd.json_normalize(data=json.load(file)["timetable"], record_path="slots", meta="day")

# Initialize facial recognition
@st.cache_resource
def init_facial_recognition():
    sfr = SimpleFacerec()
    sfr.load_encoding_images("images/")  # Load images from the `images` folder
    return sfr

# Mark attendance API call
def mark_attendance_api(student_id, subject):
    url = "http://localhost:5001/mark-attendance"
    payload = {"student_id": student_id, "subject": subject}
    response = requests.post(url, json=payload)
    return response.json()

# Dashboard Page
def dashboard():
    sfr = init_facial_recognition()  # Initialize face recognition
    st.title("Student Dashboard")

    # Load the timetable
    timetable = load_timetable()
    student_id = st.text_input("Enter Student ID", value="12345")

    # Display timetable by day
    days = timetable["day"].unique()
    selected_day = st.selectbox("Select Day", days)

    filtered_timetable = timetable[timetable["day"] == selected_day]

    st.subheader(f"Timetable for {selected_day}")

    # Style for the attendance button
    attendance_button_style = """
        <style>
        .attendance-btn {
            background-color: #4CAF50; /* Green */
            border: none;
            color: white;
            padding: 12px 25px; /* Increased padding for better spacing */
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 14px;
            margin: 10px 5px;
            border-radius: 8px;
            cursor: pointer;
        }
        .attendance-btn:hover {
            background-color: #45a049;
        }
        </style>
    """
    st.markdown(attendance_button_style, unsafe_allow_html=True)

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
            with cols[0]:  # Time
                st.markdown(
                    f"<div style='background-color:{bg_color}; padding:15px; border-radius:8px; text-align:center; font-weight:bold;'>{time}</div>",
                    unsafe_allow_html=True,
                )
            with cols[1]:  # Subject
                st.markdown(
                    f"<div style='background-color:{bg_color}; padding:15px; border-radius:8px; font-weight:bold;'>{subject}</div>",
                    unsafe_allow_html=True,
                )
            with cols[2]:  # Teacher
                if subject == "Break":
                    st.markdown(
                        f"<div style='background-color:{bg_color}; padding:15px; border-radius:8px; text-align:center;'>Break</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<div style='background-color:{bg_color}; padding:15px; border-radius:8px; text-align:center;'>{teacher}</div>",
                        unsafe_allow_html=True,
                    )
            with cols[3]:  # Attendance Button
                # Only show button for non-break slots
                if subject != "Break":
                    button_html = f"""
                    <button class="attendance-btn" id="mark-attendance-button" onclick="mark_attendance('{subject}')">{f"Mark Attendance for {subject}"}</button>
                    """
                    st.markdown(button_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)  # Add extra spacing after the timetable

# Facial Recognition Page
def facial_recognition_page():
    st.title("Facial Recognition")

    # Ensure required session state variables are available
    if "student_id" not in st.session_state or "subject" not in st.session_state:
        st.error("Missing required data. Please go back to the dashboard.")
        return

    student_id = st.session_state["student_id"]
    subject = st.session_state["subject"]
    sfr = init_facial_recognition()  # Initialize face recognition

    st.write(f"Starting facial recognition for **{subject}**...")
    st.write("Accessing webcam...")

    # Webcam and facial recognition logic
    cap = cv2.VideoCapture(0)

    recognized_name = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access the webcam.")
            break

        # Detect faces
        face_locations, face_names = sfr.detect_known_faces(frame)

        # Display results
        for face_loc, name in zip(face_locations, face_names):
            top, right, bottom, left = face_loc
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            if name != "Unknown":
                recognized_name = name

        # Display frame in Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame, channels="RGB", use_column_width=True)

        if recognized_name:
            st.success(f"Face recognized as {recognized_name}")
            st.write("Marking attendance...")
            cap.release()

            # Call the API to mark attendance
            response = mark_attendance_api(student_id, subject)
            if response.get("message"):
                st.success(response["message"])
            else:
                st.error("Failed to mark attendance. Check backend.")
            return

    cap.release()
    st.error("No face recognized. Please try again.")

# Page Routing
if "page" not in st.session_state:
    st.session_state["page"] = "dashboard"

# Handle page transitions
if st.session_state["page"] == "dashboard":
    dashboard()
elif st.session_state["page"] == "facial_recognition":
    facial_recognition_page()

# Function to start facial recognition (triggered by button)
def mark_attendance(subject):
    st.session_state["subject"] = subject  # Save the subject to session state
    st.session_state["page"] = "facial_recognition"  # Navigate to the facial recognition page
    st.experimental_rerun()  # Rerun the app to update the page