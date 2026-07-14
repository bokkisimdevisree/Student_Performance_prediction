import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/student_model.pkl")

print("=" * 50)
print(" STUDENT PERFORMANCE PREDICTION SYSTEM ")
print("=" * 50)

# User Input
study_hours = float(input("Enter Study Hours (1-10): "))
attendance = float(input("Enter Attendance (%): "))
assignments = float(input("Enter Assignment Score (%): "))
previous_marks = float(input("Enter Previous Exam Marks: "))
sleep_hours = float(input("Enter Sleep Hours: "))
internet_usage = float(input("Enter Internet Usage (hours/day): "))
extra_classes = int(input("Extra Classes (1=Yes, 0=No): "))
sports_hours = float(input("Sports Hours per Week: "))
family_support = int(input("Family Support (1-5): "))
screen_time = float(input("Screen Time (hours/day): "))
motivation_level = int(input("Motivation Level (1-5): "))

# Create DataFrame
new_student = pd.DataFrame({
    "Study_Hours": [study_hours],
    "Attendance": [attendance],
    "Assignments": [assignments],
    "Previous_Marks": [previous_marks],
    "Sleep_Hours": [sleep_hours],
    "Internet_Usage": [internet_usage],
    "Extra_Classes": [extra_classes],
    "Sports_Hours": [sports_hours],
    "Family_Support": [family_support],
    "Screen_Time": [screen_time],
    "Motivation_Level": [motivation_level]
})

# Predict
predicted_marks = model.predict(new_student)[0]

# Keep marks between 0 and 100
predicted_marks = max(0, min(100, predicted_marks))

# Assign Grade
if predicted_marks >= 90:
    grade = "A+"
elif predicted_marks >= 80:
    grade = "A"
elif predicted_marks >= 70:
    grade = "B"
elif predicted_marks >= 60:
    grade = "C"
elif predicted_marks >= 50:
    grade = "D"
else:
    grade = "F"

print("\n" + "=" * 50)
print(f"Predicted Final Marks : {predicted_marks:.2f}")
print(f"Performance Grade     : {grade}")
print("=" * 50)