import pandas as pd

# Load dataset
df = pd.read_csv("data/student_performance.csv")

# Display basic information
print("Dataset Shape:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nData Types:")
print(df.dtypes)

# Remove duplicates if any
df = df.drop_duplicates()

# Save cleaned dataset
df.to_csv("data/cleaned_student_performance.csv", index=False)

print("\nPreprocessing Completed Successfully!")
print("Cleaned Dataset Shape:", df.shape)