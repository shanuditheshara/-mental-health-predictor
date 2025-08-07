import streamlit as st
import pandas as pd
import pickle

# Load model and encoders
with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

st.set_page_config(page_title="Mental Health Predictor", layout="centered")
st.title("🧠 Mental Health Predictor")

with st.form("mental_health_form"):
    profession = st.selectbox("Profession", label_encoders["Profession"].classes_.tolist())
    role = st.selectbox("Working Professional or Student", label_encoders["Working Professional or Student"].classes_.tolist())
    study_satisfaction = st.slider("Study Satisfaction (0-10)", 0, 10, 5)
    job_satisfaction = st.slider("Job Satisfaction (0-10)", 0, 10, 5)
    sleep_duration = st.selectbox("Sleep Duration", ["Less than 5 hours", "5-6 hours", "6-7 hours", "7-8 hours", "More than 8 hours"])
    dietary_habits = st.selectbox("Dietary Habits", label_encoders["Dietary Habits"].classes_.tolist())
    degree = st.selectbox("Degree", label_encoders["Degree"].classes_.tolist())
    financial_stress = st.slider("Financial Stress (1-10)", 1, 10, 5)
    family_history = st.selectbox("Family History of Mental Illness", label_encoders["Family History of Mental Illness"].classes_.tolist())
    work_hours = st.selectbox("Work/Study Hours", ["Less than 5 hours", "5-6 hours", "6-7 hours", "7-8 hours", "More than 8 hours"])
    academic_pressure = st.slider("Academic Pressure (0-10)", 0, 10, 5)
    work_pressure = st.slider("Work Pressure (0-10)", 0, 10, 5)

    submitted = st.form_submit_button("Predict")

if submitted:
    def convert_hours(val):
        mapping = {
            "Less than 5 hours": 4,
            "5-6 hours": 5.5,
            "6-7 hours": 6.5,
            "7-8 hours": 7.5,
            "More than 8 hours": 9
        }
        return mapping[val]

    sleep_numeric = convert_hours(sleep_duration)
    work_numeric = convert_hours(work_hours)

    input_data = {
        "Profession": profession,
        "Working Professional or Student": role,
        "Study Satisfaction": study_satisfaction,
        "Job Satisfaction": job_satisfaction,
        "Sleep Duration": sleep_numeric,
        "Dietary Habits": dietary_habits,
        "Degree": degree,
        "Financial Stress": financial_stress,
        "Family History of Mental Illness": family_history,
        "Work/Study Hours": work_numeric,
        "Academic Pressure": academic_pressure,
        "Work Pressure": work_pressure
    }

    try:
        # Encode categorical inputs
        for col in label_encoders:
            input_data[col] = label_encoders[col].transform([input_data[col]])[0]

        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        st.success(f"🎯 Predicted Mental Health Outcome: **{prediction}**")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
