
import streamlit as st
import numpy as np
import pickle

model = pickle.load(open('heart_disease_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("Heart Disease Prediction System")

age = st.number_input("Age", min_value=1)
gender = st.selectbox("Gender", [1, 2])
height = st.number_input("Height")
weight = st.number_input("Weight")
ap_hi = st.number_input("AP_HI")
ap_lo = st.number_input("AP_LO")
cholesterol = st.selectbox("Cholesterol", [1, 2, 3])
glucose = st.selectbox("Glucose", [1, 2, 3])
smoke = st.selectbox("Smoke", [0, 1])
alco = st.selectbox("Alcohol", [0, 1])
active = st.selectbox("Active", [0, 1])

if st.button("Predict"):
    data = np.array([[age, gender, height, weight,
                      ap_hi, ap_lo, cholesterol,
                      glucose, smoke, alco, active]])

    data = scaler.transform(data)
    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("Heart Disease Detected")
    else:
        st.success("No Heart Disease Detected")
