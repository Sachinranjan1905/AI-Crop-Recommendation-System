import streamlit as st
import joblib
import numpy as np
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from PIL import Image


st.set_page_config(page_title="AI Crop Recommendation", page_icon="🌱", layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_dir = os.path.join(BASE_DIR, "model")
model_path = os.path.join(model_dir, "crop_model.pkl")
data_path = os.path.join(BASE_DIR, "data", "Crop_recommendation.csv")

# Create model folder if not exists
if not os.path.exists(model_dir):
    os.makedirs(model_dir)


# Train model if not exists
if not os.path.exists(model_path):

    st.warning("Model not found. Training model automatically...")

    data = pd.read_csv(data_path)

    X = data.drop("label", axis=1)
    y = data["label"]

    model = RandomForestClassifier()
    model.fit(X, y)

    joblib.dump(model, model_path)

    st.success("Model trained successfully!")

else:
    model = joblib.load(model_path)


st.title("🌱 AI Crop Recommendation System")
st.write("Enter Soil and Weather Conditions")


# Sidebar Inputs
st.sidebar.header("Soil Parameters")

N = st.sidebar.number_input("Nitrogen (N)", 0, 200)
P = st.sidebar.number_input("Phosphorus (P)", 0, 200)
K = st.sidebar.number_input("Potassium (K)", 0, 200)

temperature = st.sidebar.number_input("Temperature (°C)", 0.0, 50.0)
humidity = st.sidebar.number_input("Humidity (%)", 0.0, 100.0)
ph = st.sidebar.number_input("pH Value", 0.0, 14.0)
rainfall = st.sidebar.number_input("Rainfall (mm)", 0.0, 500.0)


if st.sidebar.button("Predict Crop"):

    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    prediction = model.predict(features)

    st.success(f"🌾 Recommended Crop: **{prediction[0]}**")


# Model Insights
st.subheader("📊 Model Insights")

feature_graph = os.path.join(BASE_DIR, "outputs", "feature_importance.png")

if os.path.exists(feature_graph):

    image = Image.open(feature_graph)
    st.image(image, caption="Feature Importance")