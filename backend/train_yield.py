import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os

# Load dataset
data = pd.read_csv("dataset/crop_yield.csv")

# Select useful columns
data = data[
    [
        "average_rain_fall_mm_per_year",
        "pesticides_tonnes",
        "avg_temp",
        "hg/ha_yield"
    ]
]

# Define features and target
X = data.drop("hg/ha_yield", axis=1)
y = data["hg/ha_yield"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("Model trained successfully")
print("RMSE:", rmse)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/yield_model.pkl")

print("Model saved as models/yield_model.pkl")