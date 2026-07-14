import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Load cleaned dataset
df = pd.read_csv("data/cleaned_student_performance.csv")

# Features
X = df.drop(["Student_ID", "Final_Marks"], axis=1)

# Target
y = df["Final_Marks"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create models
linear_model = LinearRegression()
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train models
linear_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Predictions
linear_pred = linear_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

# Function to evaluate models
def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_true, y_pred)

    print("=" * 40)
    print(name)
    print("=" * 40)
    print(f"MAE  : {mae:.2f}")
    print(f"MSE  : {mse:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R² Score : {r2:.4f}")

    return r2

# Evaluate
linear_r2 = evaluate("Linear Regression", y_test, linear_pred)
rf_r2 = evaluate("Random Forest", y_test, rf_pred)

# Create models folder
os.makedirs("models", exist_ok=True)

# Save best model
if rf_r2 > linear_r2:
    joblib.dump(rf_model, "models/student_model.pkl")
    print("\n✅ Random Forest selected as the best model.")
else:
    joblib.dump(linear_model, "models/student_model.pkl")
    print("\n✅ Linear Regression selected as the best model.")

print("\nModel saved successfully in the models folder.")