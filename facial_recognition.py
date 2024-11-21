import streamlit as st
import pandas as pd
import cv2
import tempfile
from facial_recognition import SimpleFacerec
import json
import requests

# Load timetable
@st.cache_data
def load_timetable():
    with open("timetable.json", "r") as file:
        return pd.json_normalize(data=json.load(file)["timetable"], record_path="slots", meta="day")

# Initialize facial recognition
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

# Facial recognition function
def perform_facial_recognition(subject, sfr):
    st.title(f"Facial Recognition for {subject}")
    st.write("Starting webcam for facial recognition...")

    # Access webcam
    cap = cv2.VideoCapture(0)

    # Temporary file to store the output
    tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".avi")

    # Video writer to save output (optional)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(tmp_video.name, fourcc, 20.0, (640, 480))

    recognized_name = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to open webcam.")
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

        # Write video to file (optional)
        out.write(frame)

        # Display frame in Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame, channels="RGB", use_column_width=True)

        if recognized_name:
            st.success(f"Face recognized as {recognized_name}")
            st.write("Marking attendance...")
            cap.release()
            out.release()
            return recognized_name

    cap.release()
    out.release()
    return None

# Main Streamlit app
def main():
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
    for index, row in filtered_timetable.iterrows():
        time = row["time"]
        subject = row["subject"]
        teacher = row["teacher"]

        if subject == "Break":
            st.write(f"**{time} - Break**")
        else:
            st.write(f"**{time}** - {subject} (Teacher: {teacher})")
            if st.button(f"Mark Attendance for {subject} ({time})", key=f"{subject}-{time}"):
                # Perform facial recognition
                recognized_name = perform_facial_recognition(subject, sfr)

                if recognized_name:
                    # Call API to mark attendance
                    response = mark_attendance_api(student_id, subject)
                    if response.get("message"):
                        st.success(response["message"])
                    else:
                        st.error("Failed to mark attendance. Check backend.")

if __name__ == "__main__":
    main()