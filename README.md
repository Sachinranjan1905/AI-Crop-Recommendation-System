# 🌱 AI Crop Recommendation System

An AI-powered Crop Recommendation System built using **Machine Learning (Random Forest)** and **Streamlit**.  
The system predicts the most suitable crop based on **soil nutrients and environmental conditions**.

---

# 🚀 Features

- Machine Learning model using **Random Forest**
- Predict best crop using:
  - Nitrogen (N)
  - Phosphorus (P)
  - Potassium (K)
  - Temperature
  - Humidity
  - Soil pH
  - Rainfall
- Data Visualization
- Confusion Matrix
- Classification Report
- Interactive **Streamlit Web App**

---

# 🧠 Machine Learning Model

Algorithm Used:

- **Random Forest Classifier**

Why Random Forest?

- High accuracy
- Handles large datasets
- Works well with structured agricultural data

---

# 📂 Project Structure
AI-Crop-Recommendation-System
│
├── app
│ └── app.py
│
├── data
│ └── Crop_recommendation.csv
│
├── model
│ └── crop_model.pkl
│
├── outputs
│ ├── confusion_matrix.png
│ ├── data_distribution.png
│ └── classification_report.txt
│
├── main.py
├── requirements.txt
└── README.md

---

# 📊 Dataset Features

| Feature | Description |
|------|-------------|
| N | Nitrogen content in soil |
| P | Phosphorus content in soil |
| K | Potassium content in soil |
| temperature | Temperature (°C) |
| humidity | Humidity (%) |
| ph | Soil pH |
| rainfall | Rainfall (mm) |

Target:
Crop Label

---

# ⚙️ Installation

Clone the repository
git clone https://github.com/Sachinranjan1905/AI-Crop-Recommendation-System.git

Move into the folder

cd AI-Crop-Recommendation-System

Install dependencies


---

# ▶️ Run Model Training
python main.py

This will:

- Train the ML model
- Generate graphs
- Save model
- Save confusion matrix

---

# 🌐 Run Web App
streamlit run app/app.py

Open browser:
http://localhost:8501

---

# 📈 Model Output

The project generates:

- Data Distribution Graph
- Confusion Matrix
- Classification Report
- Saved ML Model

---

# 🖥 Example Prediction

Input:
N = 36
P = 58
K = 25
Temperature = 28.6
Humidity = 59.3
pH = 8.3
Rainfall = 36.9

Output:
Recommended Crop: Rice

---

# 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Streamlit

---

# 👨‍💻 Author

**Sachin Ranjan**

B.Tech CSE  
Quantum University  

GitHub  
https://github.com/Sachinranjan1905

---

# ⭐ Support

If you like this project, please **star the repository** ⭐# AI-Crop-Recommendation-System
Sachin1905@# AI-Crop-Recommendation-System
