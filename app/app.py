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
# ---------------------------------------------------
# RESPONSIVE CSS (Times New Roman Theme)
# ---------------------------------------------------
st.markdown("""
<style>

/* Font */

html, body, [class*="css"] {
    font-family: "Times New Roman", Times, serif !important;
    color:#1f2937 !important;
}

/* Background */

.stApp{
background: linear-gradient(135deg,#e8f5e9,#c8e6c9);
}

/* Titles */

h1{
color:#1b5e20 !important;
font-weight:700;
font-size:42px;
}

h2,h3{
color:#2e7d32 !important;
}

/* Paragraph + markdown text */

p, span, label, div{
font-size:18px;
color:#1f2937 !important;
}

/* Markdown fix */

[data-testid="stMarkdownContainer"]{
color:#1f2937 !important;
}

/* Buttons */

.stButton>button{
background:#2e7d32;
color:white !important;
border:none;
border-radius:12px;
height:3em;
font-size:18px;
font-weight:600;
}

.stButton>button:hover{
background:#1b5e20;
}

/* Card container */

.block-container{
padding-top:2rem;
background:white;
border-radius:15px;
padding:25px;
box-shadow:0 4px 15px rgba(0,0,0,0.1);
}

/* INPUT FIX */

input, textarea{
color:#000000 !important;
background-color:#ffffff !important;
}

/* Number input */

.stNumberInput input{
color:#000000 !important;
background-color:#ffffff !important;
}

/* Slider label */

.stSlider label{
color:#000000 !important;
}

/* Tabs */

.stTabs [data-baseweb="tab"]{
font-size:18px;
color:#1b5e20 !important;
}

/* Mobile */

@media (max-width:768px){

h1{
font-size:30px;
}

p{
font-size:16px;
}

}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------
# LOTTIE FUNCTION
# ---------------------------------------------------
def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_ai = load_lottie(
"https://assets3.lottiefiles.com/packages/lf20_tno6cg2w.json"
)

# ---------------------------------------------------
# PROJECT ROOT PATH
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

    model = RandomForestClassifier(n_estimators=200)

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
        "Smart Agriculture powered by Machine Learning. This system evaluates soil nutrients "
        "(N, P, K), weather conditions such as temperature, humidity, and rainfall, along with "
        "soil pH levels to recommend the most suitable crop for a given environment. "
        "The goal is to assist farmers in selecting the right crop at the right time, "
        "improving productivity, reducing risk, and promoting sustainable agriculture."
    )

with col2:

    if lottie_ai:
        st_lottie(lottie_ai,height=220)

st.divider()
st.info("🌾 AI helps farmers choose the best crop based on soil and weather conditions.")
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
        probabilities = model.predict_proba(features)

        crop = prediction[0]
        confidence = np.max(probabilities)*100

        # AI RESULT CARD
        st.markdown(f"""
        <div style="
        background:#ecfdf5;
        padding:30px;
        border-radius:15px;
        border-left:8px solid #16a34a;
        box-shadow:0px 4px 10px rgba(0,0,0,0.1);
        ">

        <h2 style="color:#065f46;">🌾 Recommended Crop</h2>

        <h1 style="color:#15803d;">{crop.upper()}</h1>

        <p style="font-size:20px;"><b>📊 Confidence:</b> {confidence:.2f}%</p>

        </div>
        """, unsafe_allow_html=True)

        # Crop image
        if crop in crop_images:
            st.image(crop_images[crop],width=350)

        # Fertilizer recommendation card
        if crop in fertilizer_data:

            st.markdown(f"""
            <div style="
            background:#f0fdf4;
            padding:20px;
            border-radius:12px;
            border-left:6px solid #22c55e;
            margin-top:15px;
            ">

            <h3>🌱 Fertilizer Recommendation</h3>

            <p style="font-size:18px;">{fertilizer_data[crop]}</p>

            </div>
            """, unsafe_allow_html=True)

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
    title="Feature Importance in Crop Prediction",
    color=importance

    )

    st.plotly_chart(fig,use_container_width=True)

# ===================================================
# ABOUT TAB
# ===================================================
with tab3:

    st.subheader("About This Project")

    st.write("""

This project uses **Machine Learning** to recommend crops based on soil nutrients and environmental conditions.

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

This system helps farmers make **data-driven crop decisions**.

""")

st.divider()

st.caption("Developed by Sachin Ranjan 🚀")