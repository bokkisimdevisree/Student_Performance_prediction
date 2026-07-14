import pandas as pd
import numpy as np

# Generate the same random values every time
np.random.seed(42)

# Number of students
n = 2500

# Student IDs
student_id = range(1, n + 1)

# Features
study_hours = np.random.randint(1, 11, n)
attendance = np.random.randint(60, 101, n)
assignments = np.random.randint(50, 101, n)
previous_marks = np.random.randint(40, 100, n)
sleep_hours = np.random.randint(4, 10, n)
internet_usage = np.random.randint(1, 8, n)
extra_classes = np.random.randint(0, 2, n)
sports_hours = np.random.randint(0, 4, n)
family_support = np.random.randint(1, 6, n)
screen_time = np.random.randint(1, 8, n)
motivation_level = np.random.randint(1, 6, n)

# Final Marks (Realistic Formula)
final_marks = (
    previous_marks * 0.35 +
    study_hours * 4 +
    attendance * 0.20 +
    assignments * 0.20 +
    sleep_hours * 1.5 +
    family_support * 2 +
    motivation_level * 3 -
    internet_usage * 1.2 -
    screen_time * 1.3 +
    extra_classes * 4 +
    sports_hours * 1.5 +
    np.random.normal(0, 5, n)
)

# Keep marks between 0 and 100
final_marks = np.clip(final_marks, 0, 100)

# Create DataFrame
data = pd.DataFrame({
    "Student_ID": student_id,
    "Study_Hours": study_hours,
    "Attendance": attendance,
    "Assignments": assignments,
    "Previous_Marks": previous_marks,
    "Sleep_Hours": sleep_hours,
    "Internet_Usage": internet_usage,
    "Extra_Classes": extra_classes,
    "Sports_Hours": sports_hours,
    "Family_Support": family_support,
    "Screen_Time": screen_time,
    "Motivation_Level": motivation_level,
    "Final_Marks": final_marks.round(2)
})

# Save dataset
data.to_csv("data/student_performance.csv", index=False)

print("Dataset created successfully!")
print(data.head())
print("\nShape:", data.shape)