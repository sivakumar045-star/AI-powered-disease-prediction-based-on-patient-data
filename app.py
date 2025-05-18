import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('disease_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("AI Disease Prediction App")
st.write("Enter patient data below to predict disease risk.")

# User input
age = st.slider("Age", 20, 80, 50)
sex = st.radio("Sex", [0, 1])
cp = st.slider("Chest Pain Type (0-3)", 0, 3)
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 240)
fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.slider("Resting ECG Result (0-2)", 0, 2)
thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
exang = st.radio("Exercise-Induced Angina", [0, 1])
oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0)
slope = st.slider("Slope of Peak Exercise", 0, 2)
ca = st.slider("Major Vessels Colored", 0, 3)
thal = st.slider("Thalassemia Type", 0, 2)

# Make prediction
if st.button("Predict Disease Risk"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    if prediction[0] == 1:
        st.error("High risk of disease detected.")
    else:
        st.success("Low risk of disease.")