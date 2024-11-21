from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)

# Mock database for attendance
attendance_records = []

@app.route('/mark-attendance', methods=['POST'])
def mark_attendance():
    """
    Endpoint to mark attendance.
    Expects `student_id` and `subject` in the POST request body.
    """
    data = request.json
    student_id = data.get('student_id')
    subject = data.get('subject')
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if not student_id or not subject:
        return jsonify({"error": "Missing student_id or subject"}), 400

    # Save attendance
    attendance_records.append({
        "student_id": student_id,
        "subject": subject,
        "timestamp": timestamp
    })

    return jsonify({"message": "Attendance marked successfully!", "attendance_records": attendance_records})


if __name__ == '__main__':
    app.run(port=5001)