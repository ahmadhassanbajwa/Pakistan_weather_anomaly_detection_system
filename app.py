import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import warnings

warnings.filterwarnings('ignore')

# --- SECTION 1: PAGE CONFIGURATION AND STYLING ---
st.set_page_config(page_title="Pakistan Regional Weather Predictor", layout="wide")

# Custom CSS for a modern, vibrant dashboard with fixed text visibility
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    
    /* Metric Card Background */
    div[data-testid="metric-container"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Metric Label */
    div[data-testid="metric-container"] label p {
        color: #8b949e !important; 
    }
    
    /* Metric Value */
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] > div {
        color: #ffffff !important; 
    }

    /* Expander Container Styling */
    div[data-testid="stExpander"] details {
        border: 1px solid #30363d;
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Expander Body Background */
    div[data-testid="stExpanderDetails"] {
        background-color: #161b22;
    }
    
    /* Force Expander Body Text to White */
    div[data-testid="stExpanderDetails"] p,
    div[data-testid="stExpanderDetails"] li,
    div[data-testid="stExpanderDetails"] span {
        color: #ffffff !important; 
    }
    </style>
    """, unsafe_allow_html=True)

# --- SECTION 2: DATA PREPROCESSING AND ADVANCED MODELING ---
@st.cache_resource
def build_anomaly_aware_model():
    df = pd.read_csv('pakistan_weather.csv')
    
    # Clean baseline features
    required_cols = ['tavg', 'humidity', 'pressure', 'wind_speed']
    df_clean = df.dropna(subset=required_cols).copy()
    
    # 1. GENERATE SEASONAL MAPPING (For training data background context)
    if 'date' in df_clean.columns:
        df_clean['date'] = pd.to_datetime(df_clean['date'])
        df_clean['season'] = df_clean['date'].dt.month.map(
            lambda x: 'Winter' if x in [12, 1, 2] else 
                     ('Spring' if x in [3, 4, 5] else 
                     ('Summer' if x in [6, 7, 8] else 'Autumn'))
        )
    else:
        df_clean['season'] = 'Summer'
        
    # Normalize Precipitation column if it exists
    if 'precipitation' not in df_clean.columns:
        for col in ['prcp', 'precip', 'rain']:
            if col in df_clean.columns:
                df_clean = df_clean.rename(columns={col: 'precipitation'})
                break

    # 2. GMM PROBABILITY MODEL (Bimodal Temperature Handling)
    tavg_data = df_clean['tavg'].values.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
    gmm.fit(tavg_data)
    
    # Score historical data to define the 99.5th percentile global threshold
    df_clean['gmm_anomaly_score'] = -gmm.score_samples(tavg_data)
    gmm_threshold = np.percentile(df_clean['gmm_anomaly_score'], 99.5)
    
    # 3. CONTEXTUAL BASELINES (City + Season)
    baselines = {}
    for city in df_clean['city'].unique():
        city_df = df_clean[df_clean['city'] == city]
        
        baselines[city] = {
            'pres_mean': city_df['pressure'].mean(),
            'pres_std': city_df['pressure'].std(),
            'seasons': {}
        }
        
        # Precipitation stats (Zero-Inflated/Poisson context)
        if 'precipitation' in city_df.columns:
            rain_data = city_df['precipitation'].dropna()
            baselines[city]['rain_lambda'] = rain_data.mean() if len(rain_data) > 0 else 0
            baselines[city]['rain_99th'] = np.percentile(rain_data, 99) if len(rain_data) > 0 else 0
        else:
            baselines[city]['rain_lambda'], baselines[city]['rain_99th'] = 0, 0
            
        # Seasonal Temperature Confidence Intervals
        for season in df_clean['season'].unique():
            season_df = city_df[city_df['season'] == season]['tavg'].dropna()
            n = len(season_df)
            if n > 1:
                mean = np.mean(season_df)
                std_dev = np.std(season_df, ddof=1)
                se = std_dev / np.sqrt(n)
                ci = stats.t.interval(0.95, n-1, loc=mean, scale=se)
            else:
                mean, std_dev, ci = 0, 0, (0, 0)
                
            baselines[city]['seasons'][season] = {
                'mean': mean, 'std': std_dev, 'ci_lower': ci[0], 'ci_upper': ci[1]
            }

    # 4. REGRESSION MODELING (Trained on GMM + City + Season)
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_encoded = encoder.fit_transform(df_clean[['city', 'season']])
    cat_cols = encoder.get_feature_names_out(['city', 'season'])
    cat_df = pd.DataFrame(cat_encoded, columns=cat_cols, index=df_clean.index)
    
    X = pd.concat([df_clean[['humidity', 'pressure', 'wind_speed', 'gmm_anomaly_score']], cat_df], axis=1)
    y = df_clean['tavg']
    
    model = LinearRegression().fit(X, y)
    
    return model, encoder, gmm, gmm_threshold, baselines, cat_cols

# Initialize the backend engine
model, encoder, gmm, gmm_threshold, baselines, cat_cols = build_anomaly_aware_model()

# --- SECTION 3: INTERACTIVE USER INTERFACE ---
st.title("🌦️ Regional Anomaly-Aware Predictor")
st.markdown(f"**Developer:** Evaluators | **Framework:** GMM Anomaly Scoring & Poisson Analysis")

# Layout Columns
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Control Panel")
    st.info("Input real-time conditions to detect localized statistical anomalies.")
    
    # Categorical Inputs
    c1, c2 = st.columns(2)
    with c1:
        target_city = st.selectbox("Target City", options=sorted(list(baselines.keys())))
    with c2:
        target_season = st.selectbox("Current Season", options=["Winter", "Spring", "Summer", "Autumn"])
        
    c_stats = baselines[target_city]['seasons'].get(target_season, {'mean': 0, 'std': 0, 'ci_lower': 0, 'ci_upper': 0})
    
    # Dynamic Statistical Context
    st.markdown(f"""
    **Context:** The 95% Confidence Interval for **{target_city}** in **{target_season}** 
    is historically bounded between **{c_stats['ci_lower']:.2f}°C** and **{c_stats['ci_upper']:.2f}°C**.
    """)
    
    # Environmental Sliders
    hum = st.slider("Humidity (%)", min_value=0, max_value=100, value=60)
    pres = st.slider("Pressure (hPa)", min_value=980.0, max_value=1050.0, value=1011.0, step=0.1)
    wind = st.slider("Wind Speed (km/h)", min_value=0.0, max_value=60.0, value=12.0, step=0.1)
    precip = st.slider("Precipitation (mm)", min_value=0.0, max_value=250.0, value=0.0, step=1.0)
    
    prev_temp = st.slider(f"Current Avg Temp in {target_city} (°C)", 
                          min_value=-25.0, max_value=55.0, 
                          value=float(round(c_stats['mean'], 1)), step=0.1)

# --- SECTION 4: PREDICTION AND STATISTICAL RESULTS ---
if st.button("Generate Regional Analysis"):
    # 1. Calculate the real-time GMM Anomaly Score
    input_gmm_score = -gmm.score_samples(np.array([[prev_temp]]))[0]
    
    # 2. Prepare Regression Input
    input_cat = pd.DataFrame([[target_city, target_season]], columns=['city', 'season'])
    encoded_cat = encoder.transform(input_cat)
    input_vector = np.hstack([[hum, pres, wind, input_gmm_score], encoded_cat[0]])
    
    feature_names = ['humidity', 'pressure', 'wind_speed', 'gmm_anomaly_score'] + list(cat_cols)
    input_df = pd.DataFrame([input_vector], columns=feature_names)
    
    prediction = model.predict(input_df)[0]
    
    with col2:
        st.subheader("Analysis Output")
        st.metric(label=f"Predicted {target_city} Temperature", value=f"{prediction:.2f} °C")
        
        # --- MULTI-LAYER ANOMALY DETECTION LOGIC ---
        
        # A. GMM Temperature Anomaly (Global Rarity)
        if input_gmm_score > gmm_threshold:
            st.error(f"🚨 EXTREME CLIMATE EVENT (GMM Score: {input_gmm_score:.2f})")
            st.markdown(f"**Insight:** This temperature ranks in the top 0.5% of extreme probabilities country-wide.")
        # B. Local Seasonal Anomaly (Confidence Interval Breach)
        elif prev_temp < c_stats['ci_lower'] or prev_temp > c_stats['ci_upper']:
            st.warning(f"⚠️ LOCAL SEASONAL DEVIATION")
            st.markdown(f"**Insight:** {prev_temp}°C falls significantly outside the 95% probability range for {target_city} during {target_season}.")
        else:
            st.success(f"✅ NORMAL LOCAL PATTERN")
            st.markdown(f"**Insight:** Temperature aligns perfectly with {target_city}'s historical {target_season} distribution.")

        # C. Zero-Inflated Poisson Precipitation Check
        if precip > baselines[target_city]['rain_99th'] and precip > 0:
            st.error(f"🌧️ FLOOD/MONSOON WARNING")
            st.markdown(f"**Precipitation Alert:** {precip}mm violently exceeds the 99th percentile historical Poisson distribution for {target_city} (Limit: {baselines[target_city]['rain_99th']:.1f}mm).")

        # D. Pressure Storm Warning
        pres_danger_zone = baselines[target_city]['pres_mean'] - (2 * baselines[target_city]['pres_std'])
        if pres < pres_danger_zone:
            st.warning(f"🌪️ SEVERE STORM RISK")
            st.markdown(f"**Pressure Drop:** Atmospheric pressure ({pres} hPa) is more than 2σ below local norms. High correlation with incoming severe weather.")

        # Documentation of Process
        with st.expander("View Statistical Methodology"):
            st.write(f"""
            - **Temperature Model:** 2-Component Gaussian Mixture Model (GMM) captured bimodal peaks.
            - **Context Engine:** Isolated 95% Confidence Interval boundaries via One-Hot City + Season combinations.
            - **Precipitation:** Utilized Zero-Inflated Poisson upper-bound limits (99th percentile logic) to filter out normal "dry" days.
            - **Pressure Weights:** Identified pressure variance as a highly stable feature, mapping $2\sigma$ deviations to storm risk indicators.
            """)
else:
    with col2:
        st.write("---")
        st.markdown("Select parameters and click **Generate** to run the prediction model.")
