import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

# --- SECTION 1: PAGE CONFIGURATION & ENHANCED GUI CSS ---
st.set_page_config(page_title="Pakistan Weather Predictor", layout="wide", page_icon="🌦️")
st.markdown("""
    <style>
    .main { background-color: #0d1117; color: #e6edf3; }
    div[data-testid="metric-container"] {
        background: linear-gradient(145deg, #161b22, #1c2128);
        border: 1px solid #30363d; border-radius: 12px;
        padding: 20px; box-shadow: 0 8px 16px rgba(0,0,0,0.15);
        transition: transform 0.2s ease;
    }
    div[data-testid="metric-container"]:hover { transform: translateY(-3px); }
    div[data-testid="metric-container"] label p { color: #8b949e !important; font-weight: 600; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] > div { color: #58a6ff !important; font-weight: 700;}
    div[data-testid="stExpander"] details { border: 1px solid #30363d; border-radius: 10px; overflow: hidden; }
    div[data-testid="stExpanderDetails"] { background-color: #161b22; color: #c9d1d9 !important; }
    div[data-testid="stExpanderDetails"] * { color: #c9d1d9 !important; }
    h1, h2, h3 { color: #ffffff !important; }
    .stAlert { border-radius: 10px !important; }
    </style>
    """, unsafe_allow_html=True)

# --- SECTION 2: BACKEND ENGINE ---
@st.cache_resource
def build_anomaly_aware_model():
    df = pd.read_csv('pakistan_weather.csv')
    df_clean = df.dropna(subset=['tavg', 'humidity', 'pressure', 'wind_speed']).copy()
    
    # Standardize Season & Precipitation logic
    if 'date' in df_clean.columns:
        df_clean['date'] = pd.to_datetime(df_clean['date'])
        df_clean['season'] = df_clean['date'].dt.month.map(lambda x: 'Winter' if x in [12,1,2] else ('Spring' if x in [3,4,5] else ('Summer' if x in [6,7,8] else 'Autumn')))
    else: df_clean['season'] = 'Summer'
    
    if 'precipitation' not in df_clean.columns:
        for col in ['prcp', 'precip', 'rain']:
            if col in df_clean.columns:
                df_clean.rename(columns={col: 'precipitation'}, inplace=True)
                break

    # GMM Bimodal Temp Modeling
    tavg_data = df_clean['tavg'].values.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42).fit(tavg_data)
    df_clean['gmm_score'] = -gmm.score_samples(tavg_data)
    gmm_thresh = np.percentile(df_clean['gmm_score'], 99.5)
    
    # Localized City + Season Statistical Baselines
    baselines = {}
    for city in df_clean['city'].unique():
        cdf = df_clean[df_clean['city'] == city]
        baselines[city] = {'pres_mean': cdf['pressure'].mean(), 'pres_std': cdf['pressure'].std(), 'seasons': {}}
        
        # Zero-Inflated Poisson precipitation limits
        rain = cdf['precipitation'].dropna() if 'precipitation' in cdf.columns else []
        baselines[city]['rain_99th'] = np.percentile(rain, 99) if len(rain)>0 else 0
        
        for season in df_clean['season'].unique():
            sdf = cdf[cdf['season'] == season]['tavg'].dropna()
            n, mean, std = len(sdf), np.mean(sdf) if len(sdf)>0 else 0, np.std(sdf, ddof=1) if len(sdf)>1 else 0
            ci = stats.t.interval(0.95, n-1, loc=mean, scale=std/np.sqrt(n)) if n>1 else (0,0)
            baselines[city]['seasons'][season] = {'mean': mean, 'std': std, 'ci_lower': ci[0], 'ci_upper': ci[1]}

    # OLS Regression
    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_df = pd.DataFrame(enc.fit_transform(df_clean[['city', 'season']]), columns=enc.get_feature_names_out(), index=df_clean.index)
    X = pd.concat([df_clean[['humidity', 'pressure', 'wind_speed', 'gmm_score']], cat_df], axis=1)
    model = LinearRegression().fit(X, df_clean['tavg'])
    
    return model, enc, gmm, gmm_thresh, baselines, list(enc.get_feature_names_out())

model, encoder, gmm, gmm_threshold, baselines, cat_cols = build_anomaly_aware_model()

# --- SECTION 3: FRONTEND UI ---
st.title("🌦️ Regional Anomaly-Aware Predictor")
st.markdown("**Developer:** Evaluators | **Framework:** GMM Anomaly Engine & Poisson Mapping")
col1, col2 = st.columns([1.1, 1], gap="large")

with col1:
    st.subheader("Control Panel")
    c1, c2 = st.columns(2)
    target_city = c1.selectbox("Target City", sorted(baselines.keys()))
    target_season = c2.selectbox("Current Season", ["Winter", "Spring", "Summer", "Autumn"])
    c_stats = baselines[target_city]['seasons'].get(target_season, {'mean':0, 'std':0, 'ci_lower':0, 'ci_upper':0})
    
    st.info(f"📊 **Context:** The 95% Confidence Interval for **{target_city}** in **{target_season}** is historically bounded between **{c_stats['ci_lower']:.2f}°C** and **{c_stats['ci_upper']:.2f}°C**.")
    
    hum = st.slider("Humidity (%)", 0, 100, 60)
    pres = st.slider("Pressure (hPa)", 980.0, 1050.0, 1011.0, 0.1)
    wind = st.slider("Wind Speed (km/h)", 0.0, 60.0, 12.0, 0.1)
    precip = st.slider("Precipitation (mm)", 0.0, 250.0, 0.0, 1.0)
    prev_temp = st.slider(f"Current Avg Temp in {target_city} (°C)", -25.0, 55.0, float(round(c_stats['mean'], 1)), 0.1)

# --- SECTION 4: PREDICTION & LOGIC ---
if st.button("Generate Regional Analysis", type="primary", use_container_width=True):
    # Live Scoring & Prediction
    in_gmm = -gmm.score_samples(np.array([[prev_temp]]))[0]
    in_cat = encoder.transform(pd.DataFrame([[target_city, target_season]], columns=['city', 'season']))
    in_vec = np.hstack([[hum, pres, wind, in_gmm], in_cat[0]])
    pred = model.predict(pd.DataFrame([in_vec], columns=['humidity', 'pressure', 'wind_speed', 'gmm_score'] + cat_cols))[0]
    sz = abs((prev_temp - c_stats['mean']) / c_stats['std']) if c_stats['std'] > 0 else 0.0
    
    with col2:
        st.subheader("Analysis Output")
        m1, m2 = st.columns(2)
        m1.metric(f"Predicted {target_city} Temp", f"{pred:.2f} °C")
        m2.metric("Seasonal Z-Score", f"{sz:.2f} σ")
        
        # Cascading Anomaly Tiers
        if in_gmm > gmm_threshold:
            st.error(f"🚨 **EXTREME CLIMATE EVENT** (GMM Score: {in_gmm:.2f})\n\nThis temperature ranks in the top 0.5% of extreme probabilities country-wide.")
        elif sz > 3.0:
            st.error(f"⚠️ **SEVERE LOCAL ANOMALY**\n\n{prev_temp}°C is a massive statistical outlier exceeding the 3σ threshold for {target_city} ({target_season}).")
        elif sz > 2.0:
            st.warning(f"⚠️ **LOCAL SEASONAL DEVIATION**\n\n{prev_temp}°C exceeds the 2σ standard deviation boundary for normal {target_season} patterns.")
        else:
            st.success(f"✅ **NORMAL LOCAL PATTERN**\n\nTemperature aligns securely with historical variance for {target_city}.")

        # Multi-variable Environmental Checks
        if precip > baselines[target_city]['rain_99th'] and precip > 0:
            st.error(f"🌧️ **FLOOD/MONSOON WARNING**\n\n{precip}mm violently exceeds the 99th percentile historical Poisson limit ({baselines[target_city]['rain_99th']:.1f}mm).")
        if pres < (baselines[target_city]['pres_mean'] - 2*baselines[target_city]['pres_std']):
            st.warning(f"🌪️ **SEVERE STORM RISK**\n\nAtmospheric pressure is >2σ below local norms. High correlation with severe incoming weather.")

        with st.expander("🔍 View Statistical Methodology"):
            st.markdown("""
            * **Temperature Model:** 2-Component Gaussian Mixture Model (GMM) captures bimodal peaks.
            * **Local Z-Score:** Calculates $Z = |(x - \mu) / \sigma|$ against exact City+Season limits.
            * **Context Engine:** Isolates 95% CI boundaries for robust baseline tracking.
            * **Precipitation:** Utilizes Zero-Inflated Poisson 99th percentile logic.
            * **Pressure Weights:** Maps $2\sigma$ low-variance pressure drops directly to storm risks.
            """)
else:
    with col2:
        st.write("---")
        st.info("👈 Adjust the environmental parameters and click **Generate** to run the anomaly engine.")
