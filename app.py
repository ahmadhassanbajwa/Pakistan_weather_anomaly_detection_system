import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

# 1. Page Config for a Vibrant Look
st.set_page_config(page_title="Pakistan Weather Predictor", layout="wide")

# 2. Data & Model Logic (Backend)
@st.cache_resource
def get_model():
    df = pd.read_csv('pakistan_weather_clean_final.csv')
    m, s = df['tavg'].mean(), df['tavg'].std()
    df['tavg_zscore'] = np.abs((df['tavg'] - m) / s)
    X = df[['humidity', 'pressure', 'wind_speed', 'tavg_zscore']]
    y = df['tavg']
    model = LinearRegression().fit(X, y)
    return model, m, s

model, t_mean, t_std = get_model()

# 3. Vibrant UI (Frontend)
st.title("🌦️ Weather Anomaly Dashboard")
st.write("Developed by Evaluators")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Environmental Parameters")
    hum = st.slider("Humidity (%)", 0, 100, 60)
    pres = st.number_input("Pressure (hPa)", value=1011.0)
    wind = st.number_input("Wind Speed (km/h)", value=12.0)
    prev = st.number_input("Prev. Temp (°C)", value=22.0)

if st.button("Predict with Anomaly Analysis"):
    z = abs((prev - t_mean) / t_std)
    features = np.array([[hum, pres, wind, z]])
    pred = model.predict(features)[0]
    
    with col2:
        st.subheader("Prediction Result")
        st.metric("Predicted Temperature", f"{pred:.2f} °C")
        if z > 2:
            st.error(f"Anomaly Detected! (Z-Score: {z:.2f})")
        else:
            st.success("Pattern is Normal")
