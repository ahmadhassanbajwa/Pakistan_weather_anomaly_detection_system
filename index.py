from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

app = FastAPI()

# --- BACKEND LOGIC: Load data once on startup ---
df = pd.read_csv('pakistan_weather_clean_final.csv')
t_mean, t_std = df['tavg'].mean(), df['tavg'].std()
df['tavg_zscore'] = np.abs((df['tavg'] - t_mean) / t_std)

X = df[['humidity', 'pressure', 'wind_speed', 'tavg_zscore']]
y = df['tavg']
model = LinearRegression().fit(X, y)

class WeatherInput(BaseModel):
    humidity: float
    pressure: float
    wind_speed: float
    prev_temp: float

@app.post("/api/predict")
async def predict(data: WeatherInput):
    # Calculate anomaly part in real-time
    zscore = abs((data.prev_temp - t_mean) / t_std)
    
    # Predict using the trained model
    features = np.array([[data.humidity, data.pressure, data.wind_speed, zscore]])
    prediction = model.predict(features)[0]
    
    return {
        "prediction": round(prediction, 2),
        "is_anomaly": zscore > 2,
        "zscore": round(zscore, 2)
    }