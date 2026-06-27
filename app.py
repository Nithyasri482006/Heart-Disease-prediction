import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("heart_disease_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")

st.title("❤️ Heart Disease Prediction System")

st.write("Enter the patient's details below.")

# --------------------------
# User Inputs
# --------------------------

age = st.number_input("Age", 20, 100, 40)

sex = st.selectbox("Sex", ["M", "F"])

chest = st.selectbox(
    "Chest Pain Type",
    ["ATA", "NAP", "ASY", "TA"]
)

bp = st.number_input("Resting Blood Pressure", 80, 250, 120)

chol = st.number_input("Cholesterol", 0, 700, 200)

fbs = st.selectbox("Fasting Blood Sugar >120 mg/dl", [0, 1])

ecg = st.selectbox(
    "Resting ECG",
    ["Normal", "ST", "LVH"]
)

maxhr = st.number_input("Maximum Heart Rate", 60, 220, 150)

angina = st.selectbox(
    "Exercise Induced Angina",
    ["Y", "N"]
)

oldpeak = st.number_input(
    "Old Peak",
    min_value=0.0,
    max_value=10.0,
    value=1.0
)

slope = st.selectbox(
    "ST Slope",
    ["Up", "Flat", "Down"]
)

# --------------------------
# Prediction
# --------------------------

if st.button("Predict"):

    input_data = pd.DataFrame({
        "Age":[age],
        "Sex":[sex],
        "ChestPainType":[chest],
        "RestingBP":[bp],
        "Cholesterol":[chol],
        "FastingBS":[fbs],
        "RestingECG":[ecg],
        "MaxHR":[maxhr],
        "ExerciseAngina":[angina],
        "Oldpeak":[oldpeak],
        "ST_Slope":[slope]
    })

    # One Hot Encoding
    input_data = pd.get_dummies(input_data, dtype=int)

    # Match training columns expected by the model (15 columns due to drop_first=True during training)
    train_columns = [
        'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak', 
        'Sex_M', 
        'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA', 
        'RestingECG_Normal', 'RestingECG_ST', 
        'ExerciseAngina_Y', 
        'ST_Slope_Flat', 'ST_Slope_Up'
    ]

    input_data = input_data.reindex(columns=train_columns, fill_value=0)

    # Ensure all data is numeric (float) before scaling
    input_data = input_data.astype(float)

    # Scale
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    
    # Calculate probability percentage natively using the Random Forest model
    probabilities = model.predict_proba(input_scaled)[0]
    
    if prediction[0] == 1:
        prob_percent = probabilities[1] * 100
    else:
        prob_percent = probabilities[0] * 100

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error(f"⚠️ The patient is likely to have Heart Disease. (Confidence: {prob_percent:.1f}%)")
    else:
        st.success(f"✅ The patient is unlikely to have Heart Disease. (Confidence: {prob_percent:.1f}%)")