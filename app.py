import streamlit as st
import joblib
import pandas as pd

model = joblib.load("NV-model.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")


st.title("Heart Stroke Prediction")
st.markdown("Provide the following details to get a stroke prediction.")

age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["Male", "Female"])
chest_pain = st.selectbox(
    "Chest Pain Type",
    ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
)
restingBP = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mm Hg)", 100, 600, 200)
fastingBS = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
restingECG = st.selectbox(
    "Resting Electrocardiographic results",
    ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"],
)
maxHR = st.slider("Maximum Heart Rate (bpm)", 60, 200, 140)
exercise_angina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST SLope", ["Upsloping", "Flat", "Downsloping"])

if st.button("Predict"):
    sex_val = "M" if sex == "Male" else "F"

    if chest_pain == "Typical Angina":
        cp_val = "TA"
    elif chest_pain == "Atypical Angina":
        cp_val = "ATA"
    elif chest_pain == "Non-anginal Pain":
        cp_val = "NAP"
    else:
        cp_val = "ASY"

    fasting_val = 1 if fastingBS == "Yes" else 0

    if restingECG == "Normal":
        ecg_val = "Normal"
    elif restingECG == "ST-T wave abnormality":
        ecg_val = "ST"
    else:
        ecg_val = "LVH"

    angina_val = "Y" if exercise_angina == "Yes" else "N"

    if st_slope == "Upsloping":
        slope_val = "Up"
    elif st_slope == "Flat":
        slope_val = "Flat"
    else:
        slope_val = "Down"

    raw_input = {
        "Age": age,
        "RestingBP": restingBP,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_val,
        "MaxHR": maxHR,
        "Oldpeak": oldpeak,
        f"Sex_{sex_val}": 1,
        f"ChestPainType_{cp_val}": 1,
        f"RestingECG_{ecg_val}": 1,
        f"ExerciseAngina_{angina_val}": 1,
        f"ST_Slope_{slope_val}": 1,
    }
    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]
    
    numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.error("Risk of Heart Stroke Detected!")
    else:
        st.success("No risk of Heart Stroke detected.")
