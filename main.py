# Crop Prediction using Random Forest

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# -----------------------------
# Create folders
# -----------------------------
os.makedirs("outputs", exist_ok=True)
os.makedirs("model", exist_ok=True)


# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("data/Crop_recommendation.csv")

print("Dataset Shape:", data.shape)
print("\nFirst 5 Rows:")
print(data.head())


# -----------------------------
# Check Missing Values
# -----------------------------
print("\nMissing Values:")
print(data.isnull().sum())


# -----------------------------
# Data Visualization
# -----------------------------
numerical_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

plt.figure(figsize=(15, 10))

for i, col in enumerate(numerical_cols):
    plt.subplot(3, 3, i + 1)
    sns.histplot(data[col], kde=True)
    plt.title(col)

plt.tight_layout()
plt.savefig("outputs/data_distribution.png")
plt.close()

print("Data distribution graph saved!")


# -----------------------------
# Feature & Target Split
# -----------------------------
X = data.iloc[:, :-1]
y = data.iloc[:, -1]


# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Data Size:", X_train.shape)
print("Testing Data Size:", X_test.shape)


# -----------------------------
# Model Training
# -----------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

print("\nModel Training Completed!")


# -----------------------------
# Prediction
# -----------------------------
y_pred = model.predict(X_test)


# -----------------------------
# Accuracy
# -----------------------------
accuracy = model.score(X_test, y_test)
print("\nAccuracy:", accuracy)


# -----------------------------
# Classification Report
# -----------------------------
report = classification_report(y_test, y_pred)

print("\nClassification Report:")
print(report)

with open("outputs/classification_report.txt", "w") as f:
    f.write(report)

print("Classification report saved!")


# -----------------------------
# Confusion Matrix
# -----------------------------
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt='d',
            cmap='Blues',
            xticklabels=model.classes_,
            yticklabels=model.classes_)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("outputs/confusion_matrix.png")
plt.close()

print("Confusion matrix saved!")


# -----------------------------
# Feature Importance
# -----------------------------
importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(8,6))
sns.barplot(x=importances, y=features)

plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")

plt.tight_layout()
plt.savefig("outputs/feature_importance.png")
plt.close()

print("Feature importance graph saved!")


# -----------------------------
# Save Model
# -----------------------------
joblib.dump(model, "model/crop_model.pkl")

print("Model saved successfully!")