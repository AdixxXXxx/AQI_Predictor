import streamlit as st
import joblib
import numpy as np


model = joblib.load("aqi_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="AQI Predictor", page_icon="🌍", layout="centered")

# Title
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>🌍 Air Quality Index Predictor</h1>",
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)

# Input section
st.markdown("### Enter Pollutant Levels (µg/m³ or mg/m³):")

col1, col2 = st.columns(2)

with col1:
    pm25 = st.number_input("**PM₂.₅ (µg/m³)**", min_value=0.0, step=0.1)
    pm10 = st.number_input("**PM₁₀ (µg/m³)**", min_value=0.0, step=0.1)
    no2 = st.number_input("**NO₂ (µg/m³)**", min_value=0.0, step=0.1)
    nox = st.number_input("**NOx (µg/m³)**", min_value=0.0, step=0.1)

with col2:
    so2 = st.number_input("**SO₂ (µg/m³)**", min_value=0.0, step=0.1)
    co = st.number_input("**CO (mg/m³)**", min_value=0.0, step=0.1)
    o3 = st.number_input("**O₃ (µg/m³)**", min_value=0.0, step=0.1)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### Enter Weather Data:")

col3, col4 = st.columns(2)

with col3:
    temp = st.number_input("**Temperature (°C)**", step=0.1)
with col4:
    humidity = st.number_input("**Humidity (%RH)**", step=0.1)

# Function to categorize AQI
def categorize_aqi(aqi):
    if aqi <= 50:
        return "Good 😀", "green"
    elif aqi <= 100:
        return "Satisfactory 🙂", "lightgreen"
    elif aqi <= 200:
        return "Moderate 😐", "orange"
    elif aqi <= 300:
        return "Poor 😷", "orangered"
    elif aqi <= 400:
        return "Very Poor 🤢", "red"
    else:
        return "Severe ☠️", "darkred"

# Prediction button
st.markdown("<br>", unsafe_allow_html=True)
center_btn = st.columns([1, 2, 1])
with center_btn[1]:
    if st.button("🔍 Predict AQI", use_container_width=True):
        features = np.array([[pm25, pm10, nox, no2, so2, co, o3, temp, humidity]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]

        category, color = categorize_aqi(prediction)

        st.markdown(
            f"""
            <div style="padding:15px; border-radius:10px; text-align:center;
                        background-color:{color}; color:white; font-size:22px;">
                ✅ Predicted AQI: <b>{prediction:.2f}</b><br>
                Category: <b>{category}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )