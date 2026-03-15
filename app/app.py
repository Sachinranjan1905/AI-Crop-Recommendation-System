import streamlit as st
import joblib
import numpy as np
import os
from PIL import Image


# Page config
st.set_page_config(page_title="AI Crop Recommendation", page_icon="🌱", layout="wide")


# Get project root path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "model", "crop_model.pkl")


# Load model safely
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error("⚠ Model file not found. Please train the model first using main.py")
    st.stop()


# Title
st.title("🌱 AI Crop Recommendation System")
st.write("Enter Soil and Weather Conditions")


# Sidebar Inputs
st.sidebar.header("Soil Parameters")

N = st.sidebar.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50)
P = st.sidebar.number_input("Phosphorus (P)", min_value=0, max_value=200, value=50)
K = st.sidebar.number_input("Potassium (K)", min_value=0, max_value=200, value=50)

temperature = st.sidebar.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
humidity = st.sidebar.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
ph = st.sidebar.number_input("pH Value", min_value=0.0, max_value=14.0, value=6.5)
rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)


# Prediction Button
if st.sidebar.button("Predict Crop"):

    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    prediction = model.predict(features)

    st.success(f"🌾 Recommended Crop: **{prediction[0]}**")


# Show Model Insights
st.subheader("📊 Model Insights")

feature_graph = os.path.join(BASE_DIR, "outputs", "feature_importance.png")

if os.path.exists(feature_graph):
    image = Image.open(feature_graph)
    st.image(image, caption="Feature Importance")
else:
    st.info("Feature importance graph not found.")