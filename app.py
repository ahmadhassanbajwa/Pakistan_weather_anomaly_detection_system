import streamlit as st
import pandas as pd
import numpy as np
import os
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

# --- SECTION 1: PAGE CONFIGURATION AND STYLING ---
st.set_page_config(page_title="Pakistan Regional Weather Predictor", layout="wide")

# Custom CSS for a modern, vibrant dark-themed dashboard
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { 
        border: 1px solid #30363d; 
        border-radius: 15px; 
        padding: 20px; 
        background: #161b22; 
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }
    div[data-testid="stExpander"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SECTION 2: DATA PREPROCESSING AND REGIONAL MODELING ---
@st.cache_resource
def build_anomaly_aware_model():
    # BULLETPROOF FILE LOADING: Dynamically get the absolute path to the CSV
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, 'pakistan_weather.csv')
    
    # Load the dataset using the absolute path
    df = pd.read_csv(csv_path)
    
    # 1. TACKLING SKEWNESS: Calculate local statistics per city
    city_baselines = df.groupby('city')['tavg'].agg(['mean', 'std']).to_dict('index')
    
    # 2. FEATURE ENGINEERING: Apply Regional Normalization (Local Z-Scores)
    def get_local_z(row):
        stats = city_baselines[row['city']]
        return abs((row['tavg'] - stats['mean']) / stats['std'])
    
    df['local_anomaly_score'] = df.apply(get_local_z, axis=1)
    
    # 3. TACKLING VARIANCE: One-Hot Encoding for City Identifiers
    encoder = OneHotEncoder(sparse_output=False)
    city_encoded = encoder.fit_transform(df[['city']])
    city_cols = encoder.get_feature_names_out(['city'])
    city_df = pd.DataFrame(city_encoded, columns=city_cols)
    
    # 4. REGRESSION MODELING: features include local anomaly and city bits
    X = pd.concat([df[['humidity', 'pressure', 'wind_speed', 'local_anomaly_score']], city_df], axis=1)
    y = df['tavg']
    
    # Train the working backend model
    model = LinearRegression().fit(X, y)
    
    return model, encoder, city_baselines, city_cols

# Initializing the engine
model, encoder, city_baselines, city_cols = build_anomaly_aware_model()

# --- SECTION 3: INTERACTIVE USER INTERFACE ---
st.title("🌦️ Regional Anomaly-Aware Predictor")
st.markdown(f"**Developer:** Ahmad Hassan | **Framework:** AI-Powered Regression Analysis")

# Layout Columns
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Control Panel")
    st.info("Adjust the sliders to simulate real-time weather conditions for a specific region.")
    
    # Selection of City
    target_city = st.selectbox("Select Target City", options=sorted(list(city_baselines.keys())))
    c_stats = city_baselines[target_city]
    
    # Interactive Sliders for environmental variables
    hum = st.slider("Humidity (%)", min_value=0, max_value=100, value=60)
    pres = st.slider("Pressure (hPa)", min_value=980.0, max_value=1050.0, value=1011.0, step=0.1)
    wind = st.slider("Wind Speed (km/h)", min_value=0.0, max_value=60.0, value=12.0, step=0.1)
    
    # Temperature slider supporting negative values for cold regions like Gilgit
    prev_temp = st.slider(f"Current Avg Temp in {target_city} (°C)", 
                          min_value=-25.0, max_value=50.0, 
                          value=float(round(c_stats['mean'], 1)), step=0.1)

# --- SECTION 4: PREDICTION AND STATISTICAL RESULTS ---
if st.button("Generate Regional Analysis"):
    # 1. Calculate local anomaly intensity for the input
    local_z = abs((prev_temp - c_stats['mean']) / c_stats['std'])
    
    # 2. Encode the city selection for the model
    city_input_df = pd.DataFrame([[target_city]], columns=['city'])
    encoded_bits = encoder.transform(city_input_df)
    
    # 3. Construct the input vector matching the training features
    input_vector = np.hstack([[hum, pres, wind, local_z], encoded_bits[0]])
    
    # 4. Create DataFrame with proper feature names to avoid warnings
    feature_names = ['humidity', 'pressure', 'wind_speed', 'local_anomaly_score'] + list(city_cols)
    input_df = pd.DataFrame([input_vector], columns=feature_names)
    
    prediction = model.predict(input_df)[0]
    
    with col2:
        st.subheader("Analysis Output")
        
        # Predicted Temperature Result
        st.metric(label=f"Predicted {target_city} Temperature", value=f"{prediction:.2f} °C")
        
        # Anomaly Classification
        if local_z > 2.0:
            st.error(f"⚠️ LOCAL ANOMALY DETECTED (Z-Score: {local_z:.2f})")
            st.markdown(f"""
            **Insight:** This temperature is highly unusual for **{target_city}**. 
            It falls outside the 95% probability density for this specific city's climate.
            """)
        else:
            st.success(f"✅ NORMAL LOCAL PATTERN (Z-Score: {local_z:.2f})")
            st.markdown(f"**Insight:** This input aligns with the historical statistical distribution for **{target_city}**.")

        # Documentation of Statistical Process
        with st.expander("View Statistical Methodology"):
            st.write(f"""
            - **Distribution:** KDE analysis confirmed skewed/multimodal distributions in Pakistan.
            - **Normalization:** Used city-specific mean ({c_stats['mean']:.2f}) and std ({c_stats['std']:.2f}).
            - **Regression:** Ordinary Least Squares considering local anomaly context.
            """)
else:
    with col2:
        st.write("---")
        st.markdown("Select parameters and click **Generate** to run the prediction model.")
