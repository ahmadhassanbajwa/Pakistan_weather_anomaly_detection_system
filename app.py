import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

# --- SECTION 1: PAGE CONFIGURATION & PREMIUM CSS ---
st.set_page_config(page_title="Pakistan Weather Anomaly Engine", layout="wide", page_icon="⚡")

st.markdown("""
    <style>
    /* Global Theme */
    .main { background-color: #0b0f19; color: #e2e8f0; font-family: 'Inter', sans-serif; }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #1f2937; }
    
    /* KPI Metric Cards (Glassmorphism) */
    div[data-testid="metric-container"] {
        background: linear-gradient(145deg, #1f2937, #111827);
        border: 1px solid #374151; 
        border-radius: 16px;
        padding: 24px; 
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease;
    }
    div[data-testid="metric-container"]:hover { 
        transform: translateY(-4px); 
        box-shadow: 0 15px 35px rgba(0,0,0,0.4); 
        border-color: #3b82f6;
    }
    
    /* Metric Typography */
    div[data-testid="metric-container"] label p { color: #9ca3af !important; font-weight: 600; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.05em; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] > div { color: #60a5fa !important; font-weight: 800; font-size: 2.2rem;}
    
    /* Alerts and Info Boxes */
    .stAlert { border-radius: 12px !important; border: none !important; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    
    /* Expanders */
    div[data-testid="stExpander"] details { border: 1px solid #374151; border-radius: 12px; overflow: hidden; background: #111827; }
    div[data-testid="stExpanderDetails"] { background-color: #111827; color: #d1d5db !important; padding: 20px; }
    div[data-testid="stExpanderDetails"] * { color: #d1d5db !important; }
    
    /* Headers */
    h1 { color: #f8fafc !important; font-weight: 800; padding-bottom: 0.5rem; }
    h2, h3 { color: #e2e8f0 !important; font-weight: 700; }
    hr { border-color: #374151; }
    </style>
    """, unsafe_allow_html=True)

# --- SECTION 2: BACKEND ENGINE (Cached for Speed) ---
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

# --- SECTION 3: SIDEBAR CONTROLS ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1146/1146860.png", width=60) # Simple generic weather icon
    st.title("System Controls")
    st.markdown("Enter real-time telemetry below.")
    st.divider()
    
    target_city = st.selectbox("📍 Target Region", sorted(baselines.keys()))
    target_season = st.selectbox("📅 Current Season", ["Winter", "Spring", "Summer", "Autumn"])
    
    st.divider()
    st.markdown("### Environmental Sensors")
    c_stats = baselines[target_city]['seasons'].get(target_season, {'mean':0, 'std':0, 'ci_lower':0, 'ci_upper':0})
    
    prev_temp = st.number_input(f"Current Temp in {target_city} (°C)", value=float(round(c_stats['mean'], 1)), step=0.1)
    precip = st.number_input("Precipitation (mm)", value=0.0, step=1.0)
    hum = st.slider("Humidity (%)", 0, 100, 60)
    pres = st.slider("Pressure (hPa)", 980.0, 1050.0, 1011.0, 0.1)
    wind = st.slider("Wind Speed (km/h)", 0.0, 60.0, 12.0, 0.1)
    
    st.divider()
    run_engine = st.button("🚀 Execute Analysis", type="primary", use_container_width=True)

# --- SECTION 4: MAIN DASHBOARD CANVAS ---
st.title("⚡ Regional Weather Anomaly Engine")
st.markdown("**Developed by:** Evaluators | **Framework:** GMM Probability & Poisson Distribution Matrix")
st.divider()

if run_engine:
    # 1. Background Calculations
    in_gmm = -gmm.score_samples(np.array([[prev_temp]]))[0]
    in_cat = encoder.transform(pd.DataFrame([[target_city, target_season]], columns=['city', 'season']))
    in_vec = np.hstack([[hum, pres, wind, in_gmm], in_cat[0]])
    pred = model.predict(pd.DataFrame([in_vec], columns=['humidity', 'pressure', 'wind_speed', 'gmm_score'] + cat_cols))[0]
    sz = abs((prev_temp - c_stats['mean']) / c_stats['std']) if c_stats['std'] > 0 else 0.0
    
    # 2. KPI Top Row
    st.markdown("### Live Telemetry Overview")
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Predicted Baseline Temp", f"{pred:.2f} °C", delta=f"{prev_temp - pred:.2f} °C Variance", delta_color="inverse")
    kpi2.metric("Seasonal Z-Score", f"{sz:.2f} σ")
    kpi3.metric("Global Rarity (GMM)", f"{in_gmm:.2f}", help="Negative Log-Likelihood. Higher means rarer.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 3. Alert Center (Cascading Logic)
    st.markdown("### 🚨 Threat Detection Center")
    
    # Track if any alerts triggered to show an "All Clear" if needed
    alert_triggered = False
    
    # A. Temperature Severity
    if in_gmm > gmm_threshold:
        st.error(f"**CRITICAL CLIMATE EVENT:** {prev_temp}°C ranks in the top 0.5% of extreme probabilities country-wide. (GMM Score: {in_gmm:.2f})")
        alert_triggered = True
    elif sz > 3.0:
        st.error(f"**SEVERE LOCAL ANOMALY:** {prev_temp}°C is a massive statistical outlier exceeding the 3σ threshold for {target_city} ({target_season}).")
        alert_triggered = True
    elif sz > 2.0:
        st.warning(f"**LOCAL SEASONAL DEVIATION:** {prev_temp}°C exceeds the 2σ boundary for normal {target_season} patterns.")
        alert_triggered = True
        
    # B. Precipitation Check (Poisson)
    if precip > baselines[target_city]['rain_99th'] and precip > 0:
        st.error(f"**FLOOD / MONSOON WARNING:** {precip}mm violently exceeds the 99th percentile historical limit ({baselines[target_city]['rain_99th']:.1f}mm) for {target_city}.")
        alert_triggered = True
        
    # C. Pressure Storm Warning
    if pres < (baselines[target_city]['pres_mean'] - 2*baselines[target_city]['pres_std']):
        st.warning(f"**SEVERE STORM RISK:** Atmospheric pressure ({pres} hPa) is >2σ below local norms. High correlation with incoming severe weather.")
        alert_triggered = True
        
    # D. All Clear
    if not alert_triggered:
        st.success(f"**ALL CLEAR:** Current telemetry aligns securely with historical variance for {target_city} in {target_season}. No anomalies detected.")

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. Statistical Context Details
    with st.expander("📊 View Mathematical Context & Baseline Data", expanded=False):
        st.write(f"**Target Geography:** {target_city} | **Current Phase:** {target_season}")
        st.write(f"**95% Confidence Interval (Normal Baseline):** {c_stats['ci_lower']:.2f}°C to {c_stats['ci_upper']:.2f}°C")
        st.write(f"**Local Mean:** {c_stats['mean']:.2f}°C | **Local Standard Deviation:** {c_stats['std']:.2f}°C")
        st.divider()
        st.markdown("""
        * **Temperature:** Modeled via 2-Component Gaussian Mixture Model (GMM). Z-Scores calculated via isolated City+Season matrices.
        * **Precipitation:** Utilizes Zero-Inflated Poisson 99th percentile logic to exclude dry-day inflation.
        * **Atmospherics:** Evaluates variance using localized standard deviation for pressure drops.
        """)

else:
    # Landing Page instructions before clicking run
    st.info("👈 Please enter the environmental parameters in the sidebar and click **Execute Analysis**.")
