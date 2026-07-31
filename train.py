import kagglehub
from kagglehub import KaggleDatasetAdapter

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Dataset filename
file_path = "heart.csv"

# Load dataset
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "johnsmith88/heart-disease-dataset",
    file_path
)

print("First 5 Records:")
print(df.head())
print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nNumerical Features:")
print(df.select_dtypes(include=["int64", "float64"]).columns)
X = df.drop("target", axis=1)

y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)
joblib.dump(model, "model.pkl")

print("Model Saved Successfully!")