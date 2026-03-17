
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from streamlit_lottie import st_lottie
import requests

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Crop Recommendation",
    page_icon="🌱",
    layout="wide"
)

# ---------------------------------------------------
# RESPONSIVE CSS
# ---------------------------------------------------
st.markdown("""
<style>

.stApp{
background: linear-gradient(120deg,#e6f9e6,#ffffff);
}

h1{
color:#2E8B57;
}

.stButton>button{
background-color:#2E8B57;
color:white;
border-radius:10px;
height:3em;
width:100%;
font-size:18px;
}

@media (max-width:768px){

h1{
font-size:28px;
}

}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOTTIE FUNCTION
# ---------------------------------------------------
def load_lottie(url):
    r = requests.get(url)
    return r.json()

lottie_ai = load_lottie(
"https://assets3.lottiefiles.com/packages/lf20_tno6cg2w.json"
)

# ---------------------------------------------------
# PATHS
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_dir = os.path.join(BASE_DIR,"model")
model_path = os.path.join(model_dir,"crop_model.pkl")
data_path = os.path.join(BASE_DIR,"data","Crop_recommendation.csv")

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# ---------------------------------------------------
# TRAIN OR LOAD MODEL
# ---------------------------------------------------
if not os.path.exists(model_path):

    data = pd.read_csv(data_path)

    X = data.drop("label",axis=1)
    y = data["label"]

    model = RandomForestClassifier()
    model.fit(X,y)

    joblib.dump(model,model_path)

else:
    model = joblib.load(model_path)

# ---------------------------------------------------
# HERO SECTION
# ---------------------------------------------------
col1,col2 = st.columns([1,1])

with col1:
    st.title("🌱 AI Crop Recommendation System")
    st.write(
    "Smart Agriculture using Machine Learning. "
    "Enter soil and weather conditions to get crop recommendation."
    )

with col2:
    st_lottie(lottie_ai,height=220)

st.divider()

# ---------------------------------------------------
# TABS
# ---------------------------------------------------
tab1,tab2,tab3 = st.tabs(["🌾 Prediction","📊 Model Insights","ℹ️ About"])

# ===================================================
# PREDICTION TAB
# ===================================================
with tab1:

    st.subheader("Enter Soil and Weather Parameters")

    col1,col2,col3 = st.columns(3)

    with col1:
        N = st.number_input("Nitrogen (N)",0,200)
        temperature = st.number_input("Temperature (°C)",0.0,50.0)

    with col2:
        P = st.number_input("Phosphorus (P)",0,200)
        humidity = st.number_input("Humidity (%)",0.0,100.0)

    with col3:
        K = st.number_input("Potassium (K)",0,200)
        ph = st.number_input("Soil pH",0.0,14.0)

    rainfall = st.slider("Rainfall (mm)",0,500,100)

    fertilizer_data = {
    "rice":"Use Urea and DAP fertilizer.",
    "maize":"Use NPK 20-20-20 fertilizer.",
    "banana":"Apply potassium rich fertilizer.",
    "apple":"Use balanced NPK and organic compost.",
    "cotton":"Nitrogen rich fertilizer recommended.",
    "grapes":"Use phosphorus rich fertilizer."
    }

    crop_images = {
    "rice":"https://upload.wikimedia.org/wikipedia/commons/6/6f/Rice_plants.jpg",
    "maize":"https://upload.wikimedia.org/wikipedia/commons/0/0c/Corncobs.jpg",
    "apple":"https://upload.wikimedia.org/wikipedia/commons/1/15/Red_Apple.jpg",
    "banana":"https://upload.wikimedia.org/wikipedia/commons/4/4c/Bananas.jpg",
    "grapes":"https://upload.wikimedia.org/wikipedia/commons/b/bb/Table_grapes_on_white.jpg"
    }

    st.write("")

    if st.button("Predict Crop 🌾"):

        features = np.array([[N,P,K,temperature,humidity,ph,rainfall]])

        prediction = model.predict(features)

        crop = prediction[0]

        st.success(f"🌾 Recommended Crop: **{crop}**")

        if crop in crop_images:
            st.image(crop_images[crop],width=350)

        if crop in fertilizer_data:
            st.info(f"🌱 Fertilizer Recommendation: {fertilizer_data[crop]}")

# ===================================================
# MODEL INSIGHTS
# ===================================================
with tab2:

    st.subheader("Feature Importance")

    features = ["N","P","K","Temperature","Humidity","pH","Rainfall"]

    importance = model.feature_importances_

    fig = px.bar(
    x=features,
    y=importance,
    labels={"x":"Features","y":"Importance"},
    title="Feature Importance in Crop Prediction"
    )

    st.plotly_chart(fig,use_container_width=True)

# ===================================================
# ABOUT TAB
# ===================================================
with tab3:

    st.subheader("About This Project")

    st.write("""
This project uses **Machine Learning** to recommend crops based on
soil nutrients and environmental conditions.

### Technologies Used
- Python
- Scikit-learn
- Pandas
- Streamlit
- Plotly

### Model
Random Forest Classifier

### Input Features
- Nitrogen
- Phosphorus
- Potassium
- Temperature
- Humidity
- pH
- Rainfall

The system helps farmers make **data driven crop decisions**.
""")

st.divider()

st.caption("Developed by Sachin Ranjan 🚀")

