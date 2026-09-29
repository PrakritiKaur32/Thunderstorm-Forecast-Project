import streamlit as st
import pandas as pd
import numpy as np
import mlflow.sklearn
from feature_engineering import engineer_features

st.set_page_config(
    page_title="Thunderstorm Prediction System",
    page_icon="⛈️",
    layout="centered"
)

st.title("⛈️ Thunderstorm Forecasting Portal")
st.markdown("Adjust meteorological readings to evaluate real-time thunderstorm probability.")

st.sidebar.header("Atmospheric Inputs")

temperature = st.sidebar.slider("Temperature (°C)", 15.0, 45.0, 30.0)
dew_point = st.sidebar.slider("Dew Point (°C)", 5.0, 35.0, 22.0)
relative_humidity = st.sidebar.slider("Relative Humidity (%)", 20.0, 100.0, 72.0)
surface_pressure = st.sidebar.slider("Surface Pressure (hPa)", 970.0, 1030.0, 1005.0)
wind_speed = st.sidebar.slider("Wind Speed (km/h)", 0.0, 70.0, 25.0)
wind_shear = st.sidebar.slider("Wind Shear (m/s)", 0.0, 50.0, 18.0)
cape_index = st.sidebar.slider("CAPE Index (J/kg)", 0.0, 4000.0, 2100.0)
k_index = st.sidebar.slider("K-Index", 0.0, 50.0, 32.0)

# Form input payload
input_raw = pd.DataFrame([{
    'temperature': temperature,
    'dew_point': dew_point,
    'relative_humidity': relative_humidity,
    'surface_pressure': surface_pressure,
    'wind_speed': wind_speed,
    'wind_shear': wind_shear,
    'cape_index': cape_index,
    'k_index': k_index
}])

# Process features
processed_input = engineer_features(input_raw)

st.subheader("Current Parameter Summary")
st.dataframe(processed_input[['temperature', 'relative_humidity', 'cape_index', 'k_index', 'dew_point_depression']])

if st.button("Generate Forecast", type="primary"):
    # Rule-based calculation representation / fallback
    risk_score = (
        (cape_index / 4000) * 0.4 +
        (relative_humidity / 100) * 0.3 +
        (k_index / 50) * 0.3
    )
    
    st.markdown("---")
    if risk_score > 0.55:
        st.error(f"⚠️ **HIGH RISK**: Severe Thunderstorm Expected! (Risk Indicator: {risk_score*100:.1f}%)")
    elif risk_score > 0.35:
        st.warning(f"⚡ **MODERATE RISK**: Scattered Convective Activity Likely. (Risk Indicator: {risk_score*100:.1f}%)")
    else:
        st.success(f"✅ **LOW RISK**: Stable Atmospheric Conditions. (Risk Indicator: {risk_score*100:.1f}%)")