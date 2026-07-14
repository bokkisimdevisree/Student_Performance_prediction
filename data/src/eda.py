import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("data/cleaned_student_performance.csv")

# Create images folder if it doesn't exist
import os
os.makedirs("images", exist_ok=True)

# Set graph style
sns.set_style("whitegrid")

# -----------------------------
# 1. Final Marks Distribution
# -----------------------------
plt.figure(figsize=(8,5))
sns.histplot(df["Final_Marks"], bins=20, kde=True)
plt.title("Distribution of Final Marks")
plt.savefig("images/final_marks_distribution.png")
plt.close()

# -----------------------------
# 2. Correlation Heatmap
# -----------------------------
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.savefig("images/correlation_heatmap.png")
plt.close()

# -----------------------------
# 3. Study Hours vs Final Marks
# -----------------------------
plt.figure(figsize=(8,5))
sns.scatterplot(x="Study_Hours", y="Final_Marks", data=df)
plt.title("Study Hours vs Final Marks")
plt.savefig("images/study_vs_marks.png")
plt.close()

# -----------------------------
# 4. Attendance vs Final Marks
# -----------------------------
plt.figure(figsize=(8,5))
sns.scatterplot(x="Attendance", y="Final_Marks", data=df)
plt.title("Attendance vs Final Marks")
plt.savefig("images/attendance_vs_marks.png")
plt.close()

# -----------------------------
# 5. Boxplot of Final Marks
# -----------------------------
plt.figure(figsize=(8,5))
sns.boxplot(y=df["Final_Marks"])
plt.title("Boxplot of Final Marks")
plt.savefig("images/final_marks_boxplot.png")
plt.close()

print("EDA completed successfully!")
print("Graphs saved inside the images folder.")