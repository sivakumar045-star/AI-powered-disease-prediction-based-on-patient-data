# disease_prediction.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import streamlit as st

# Load dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/datablist/sample-csv-files/main/files/people/heart.csv"
    df = pd.read_csv(url)
    return df

# Preprocess the data
def preprocess_data(df):
    X = df.drop('target', axis=1)
    y = df['target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42), scaler

# Train the model
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Streamlit UI
def main():
    st.title("AI-Powered Disease Prediction")
    st.markdown("Predicting the likelihood of heart disease based on patient data.")

    df = load_data()
    (X_train, X_test, y_train, y_test), scaler = preprocess_data(df)
    model = train_model(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    st.subheader("Model Accuracy")
    st.write(accuracy_score(y_test, y_pred))
    st.text(classification_report(y_test, y_pred))

    # User input
    st.subheader("Predict Disease Risk")
    age = st.number_input("Age", 20, 100, step=1)
    sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
    cp = st.slider("Chest Pain Type (0-3)", 0, 3)
    trestbps = st.number_input("Resting Blood Pressure", 80, 200)
    chol = st.number_input("Serum Cholesterol", 100, 600)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True)", [0, 1])
    restecg = st.slider("Resting ECG (0-2)", 0, 2)
    thalach = st.number_input("Max Heart Rate Achieved", 60, 220)
    exang = st.selectbox("Exercise-Induced Angina (1 = Yes)", [0, 1])
    oldpeak = st.slider("ST Depression", 0.0, 6.0, step=0.1)
    slope = st.slider("Slope of the Peak (0-2)", 0, 2)
    ca = st.slider("Major Vessels Colored (0-3)", 0, 3)
    thal = st.slider("Thalassemia (0 = Normal, 1 = Fixed Defect, 2 = Reversible)", 0, 2)

    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    input_scaled = scaler.transform(input_data)

    if st.button("Predict"):
        prediction = model.predict(input_scaled)
        if prediction[0] == 1:
            st.error("High risk of disease detected. Recommend further testing.")
        else:
            st.success("Low risk of disease.")

if __name__ == '__main__':
    main()
