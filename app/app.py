import streamlit as st
import joblib
import numpy as np
import os
from PIL import Image


# Page config
st.set_page_config(page_title="AI Crop Recommendation", page_icon="🌱")


# Get project root path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "model", "crop_model.pkl")

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


# Prediction
if st.sidebar.button("Predict Crop"):

    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    prediction = model.predict(features)

    st.success(f"🌾 Recommended Crop: {prediction[0]}")


# Show Feature Importance Graph
st.subheader("Model Insights")

feature_graph = os.path.join(BASE_DIR, "outputs", "feature_importance.png")

if os.path.exists(feature_graph):

    image = Image.open(feature_graph)
    st.image(image, caption="Feature Importance")