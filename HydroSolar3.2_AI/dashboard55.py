import os
import datetime
# --- 1. GPU ENFORCEMENT & LOG SILENCING ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import altair as alt
from datetime import datetime, timedelta
import pytz
import time
import pydeck as pdk
import streamlit.components.v1 as components 

# --- AI LIBRARIES ---
import tensorflow as tf
import keras 
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler

# --- ACCESS CONTROL (1 WEEK TIMER) ---
import datetime as dt_lib
# Set your deadline (YYYY, MM, DD)
EXPIRY_DATE = dt_lib.date(2026, 3, 31) 

if dt_lib.date.today() > EXPIRY_DATE:
    st.error(f"⛔ **ACCESS EXPIRED** \n\n This dashboard was available for review until {EXPIRY_DATE}. Please contact the administrator for access.")
    st.image("https://media.giphy.com/media/xT5LMHxhOfscxPfIfm/giphy.gif") 
    st.stop()

# --- GPU CHECK ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        gpu_status = f"🟢 GPU (Accelerated): {len(gpus)} Device(s)"
    except RuntimeError as e:
        gpu_status = "🟡 GPU (Error in Config)"
else:
    gpu_status = "⚪ CPU Mode (Standard)"

# --- ASSET DATABASE ---
PLANTS = {
    "Batang Ai": {"lat": 1.1500, "lon": 111.9000, "capacity_mw": 108,  "type": "Floating Solar + Hydro", "head_m": 46},
    "Bakun":     {"lat": 2.7628, "lon": 114.0601, "capacity_mw": 2400, "type": "Main Hydro", "head_m": 205},
    "Murum":     {"lat": 2.7167, "lon": 114.6167, "capacity_mw": 944,  "type": "Hydro", "head_m": 300},
    "Baleh":     {"lat": 1.8167, "lon": 113.6833, "capacity_mw": 1285, "type": "Construction Site", "head_m": 188}
}

PANEL_BRANDS = {
    "Generic (Old Tech)": 0.18,
    "Jinko Solar (Tiger Neo)": 0.22,
    "LONGi Solar (Hi-MO 6)": 0.228,
    "Trina Solar (Vertex S+)": 0.219,
    "Canadian Solar": 0.21
}

PEAK_PRICE = 0.40  
OFF_PEAK_PRICE = 0.20
CARBON_FACTOR = 0.45 

st.set_page_config(page_title="SEB AI Ops V99", layout="wide", page_icon="🧠")

if 'last_heartbeat' not in st.session_state:
    st.session_state['last_heartbeat'] = time.time()

# --- SIDEBAR ---
st.sidebar.header("📍 Asset Selector")
selected_plant = st.sidebar.selectbox("Select Site:", list(PLANTS.keys()))
plant_data = PLANTS[selected_plant]

st.sidebar.divider()
st.sidebar.header("🧠 AI Model Settings")
st.sidebar.caption(f"**AI Hardware:** {gpu_status}") 
epochs = st.sidebar.slider("Training Epochs:", 5, 50, 10)

st.sidebar.divider()
st.sidebar.header("⚙️ Digital Twin Settings")
selected_brand = st.sidebar.selectbox("Panel Brand:", list(PANEL_BRANDS.keys()))
brand_eff = PANEL_BRANDS[selected_brand]

estimated_area = int((plant_data['capacity_mw'] * 1_000_000) / 200)
st.sidebar.caption(f"Estimated Area: {estimated_area:,} m²")
panel_area_input = st.sidebar.number_input("Total Panel Area (m²):", value=estimated_area, step=1000)

st.sidebar.divider()
st.sidebar.subheader("🔧 Simulation Controls")
efficiency_slider = st.sidebar.slider("Simulate: Panel Health (%)", 50, 100, 92)

st.sidebar.markdown("---")
st.sidebar.write("System Design & Architecture:")
st.sidebar.markdown("### **LI CHENG YAW**")
st.sidebar.caption("© 2026 SEB Innovation Lab")

# --- HELPER: SAFE FORMAT ---
def safe_val(val, default=0.0):
    if val is None: return default
    try: return float(val)
    except: return default

# --- DATA ENGINE WITH RETRY LOGIC ---
@st.cache_data(ttl=60, show_spinner=False) 
def get_weather_data(lat, lon):
    try:
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        
        url = "https://api.open-meteo.com/v1/forecast"
        params = {"latitude": lat, "longitude": lon, "hourly": "shortwave_radiation,cloud_cover,precipitation_probability", "timezone": "Asia/Singapore", "forecast_days": 2}
        
        res = session.get(url, params=params, timeout=30)
        res.raise_for_status()
        data = res.json()
        
        df = pd.DataFrame({"time": pd.to_datetime(data["hourly"]["time"]), "Solar (W/m²)": data["hourly"]["shortwave_radiation"], "Cloud Cover (%)": data["hourly"]["cloud_cover"], "Rain Probability (%)": data["hourly"]["precipitation_probability"]})
        for col in df.columns:
            if col != 'time': df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        now_local = datetime.now(pytz.timezone('Asia/Kuching')).replace(tzinfo=None)
        closest_idx = df['time'].sub(now_local).abs().idxmin()
        row = df.iloc[closest_idx]
        return df, {"Solar": float(row["Solar (W/m²)"]), "RainProb": float(row["Rain Probability (%)"]), "Cloud": float(row["Cloud Cover (%)"])}
    except Exception as e:
        return pd.DataFrame(), {"Error": str(e)}

# --- AI BRAIN ---
@st.cache_resource(show_spinner=False)
def build_and_train_lstm():
    X_train = np.random.rand(100, 3) 
    y_train = X_train[:, 0] * 0.8 + np.random.normal(0, 0.05, 100) 
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    model = Sequential([Input(shape=(1, 3)), LSTM(50, activation='relu'), Dense(1)])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=5, verbose=0)
    return model

def run_ai_simulation(df, brand_eff, health_pct, exact_area_m2, rated_capacity_mw):
    df = df.set_index('time').resample('1min').interpolate(method='linear')
    
    # --- CRITICAL FIX: CREATE 'HOUR' COLUMN FOR DISPATCH LOGIC ---
    df['Hour'] = df.index.hour 
    
    df['Price'] = np.where((df.index.hour >= 9) & (df.index.hour <= 21), PEAK_PRICE, OFF_PEAK_PRICE)
    
    features = MinMaxScaler().fit_transform(df[['Solar (W/m²)', 'Rain Probability (%)', 'Cloud Cover (%)']].values)
    preds = build_and_train_lstm().predict(features.reshape((-1, 1, 3)), verbose=0).flatten()
    
    max_solar_mw = (exact_area_m2 * brand_eff * 1000) / 1_000_000
    df['Ideal_MW'] = preds * max_solar_mw
    df['Solar_MW'] = (df['Ideal_MW'] * (health_pct / 100)).clip(lower=0)
    
    df.loc[df['Solar (W/m²)'] < 1.1, ['Solar_MW', 'Ideal_MW']] = 0.0
    df['Recoverable_MW'] = df['Ideal_MW'] - df['Solar_MW']
    
    def get_demand_factor(hour):
        if 0 <= hour < 6: return 0.4
        elif 6 <= hour < 12: return 0.4 + ((hour-6)/6)*0.5
        elif 12 <= hour < 14: return 0.9
        elif 14 <= hour < 19: return 0.95
        elif 19 <= hour < 22: return 0.8
        else: return 0.5
    
    df['Demand_MW'] = df.index.hour.map(get_demand_factor) * (rated_capacity_mw * 1.2)

    def complex_dispatch(row):
        solar_gen = row['Solar_MW']
        price = row['Price']
        solar_ratio = solar_gen / max_solar_mw if max_solar_mw > 0 else 0
        
        # Dynamic Dispatch Logic
        if solar_ratio > 0.5:
            target_mw = rated_capacity_mw * (0.10 + (solar_ratio * 0.05))
            return target_mw, "📉 SOLAR DOMINANT: RAMP DOWN", f"Target: {target_mw:.0f} MW", "Solar", "High Supply", "Conserve Water"
        elif 9 <= row['Hour'] <= 16 and solar_ratio < 0.3:
             target_mw = rated_capacity_mw * (0.80 - (solar_ratio * 0.2)) 
             return target_mw, "☁️ CLOUD TRANSIENT: RAMP UP", f"Target: {target_mw:.0f} MW", "Solar", "Intermittent Drop", "Rapid Injection"
        elif price == PEAK_PRICE:
            target_mw = rated_capacity_mw * 0.95
            return target_mw, "⚡ PEAK DEMAND: MAX HYDRO", f"Target: {target_mw:.0f} MW", "Grid", "Peak Pricing", "Max Discharge"
        else:
            target_mw = rated_capacity_mw * 0.30
            return target_mw, "🌙 NIGHT CONSERVATION: MIN FLOW", f"Target: {target_mw:.0f} MW", "Grid", "Off-Peak", "Reservoir Recharge"

    df[['Hydro_MW', 'Instruction', 'Target_Str', 'Ctx_Source', 'Ctx_State', 'Ctx_Action']] = df.apply(lambda row: pd.Series(complex_dispatch(row)), axis=1)
    
    hydro_eff = 0.90
    head = plant_data['head_m']
    g = 9.81
    df['Flow_m3s'] = (df['Hydro_MW'] * 1000) / (g * head * hydro_eff)
    max_flow = (rated_capacity_mw * 1000) / (g * head * hydro_eff)
    
    # Non-linear Gate Calculation
    df['Gate_Open_%'] = np.power((df['Flow_m3s'] / max_flow), (1/1.5)) * 100
    df['Gate_Open_%'] = df['Gate_Open_%'].clip(0, 100) 

    return df

# --- FLEET DISPATCH LOGIC ---
def calculate_fleet_dispatch(target_load, current_health):
    fleet_status = []
    total_capacity = 0
    for name, data in PLANTS.items():
        if "Construction" in data['type']: continue
        health = current_health if name == selected_plant else 98.0
        avail_cap = data['capacity_mw'] * (health / 100.0)
        fleet_status.append({"Name": name, "Max_MW": data['capacity_mw'], "Health": health, "Avail_MW": avail_cap})
        total_capacity += avail_cap

    dispatch_results = []
    allocated_total = 0
    warning = "✅ Grid Demand Met."
    if target_load > total_capacity:
        target_load = total_capacity
        warning = "⚠️ Demand exceeds Capacity!"

    for dam in fleet_status:
        share = dam['Avail_MW'] / total_capacity
        dam_target = target_load * share
        dispatch_results.append({"Dam": dam['Name'], "Dispatch (MW)": dam_target, "Health (%)": dam['Health']})
        allocated_total += dam_target
        
    return pd.DataFrame(dispatch_results), warning, allocated_total

# --- CHART HELPERS ---
def make_chart(data, y_col, color_hex, title, current_real_val=None):
    c = data.resample('30min').mean(numeric_only=True).reset_index()
    base = alt.Chart(c).encode(x=alt.X('time:T', axis=alt.Axis(format='%H:%M')), tooltip=['time:T', y_col])
    area = base.mark_area(line={'color': color_hex}, color=alt.Gradient(gradient='linear', stops=[alt.GradientStop(color=color_hex, offset=0), alt.GradientStop(color='white', offset=1)], x1=1, x2=1, y1=1, y2=0)).encode(y=alt.Y(y_col, title=title)).properties(height=350)
    
    now = datetime.now(pytz.timezone('Asia/Kuching'))
    now_val = safe_val(current_real_val)
    if current_real_val is None:
        try: now_val = data.iloc[data.index.get_indexer([now], method='nearest')[0]][y_col]
        except: now_val = 0.0

    rule_df = pd.DataFrame({'time': [now], 'value': [now_val], 'label': ['CURRENT STATUS']})
    rule = alt.Chart(rule_df).mark_rule(color='red', strokeWidth=2).encode(
        x='time:T',
        tooltip=[alt.Tooltip('label', title='Event'), alt.Tooltip('time', format='%d %b %Y %H:%M:%S'), alt.Tooltip('value', format='.2f', title=title)]
    )
    return (area + rule).interactive()

def render_iframe_header(server_timestamp):
    now_k = datetime.now(pytz.timezone('Asia/Kuching'))
    date_str_py = now_k.strftime("%A, %d %B %Y")

    # CSS: Using --text-color variable to auto-flip Black/White based on Streamlit Theme
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: sans-serif; background-color: transparent; color: var(--text-color, #00FFFF); margin: 0; padding: 10px 0; }}
            .container {{ display: flex; justify-content: space-between; align-items: flex-start; }}
            h1 {{ margin: 0; font-size: 2rem; color: var(--text-color, #00FFFF); }}
            .right-panel {{ text-align: right; }}
            .clock {{ font-size: 2rem; font-weight: 700; color: var(--text-color, #00FFFF); }}
            .sub-text {{ font-size: 0.9rem; color: var(--text-color, #00FFFF); }}
            .blink-green {{ color: #00FF00; animation: blinker 1s linear infinite; }}
            @keyframes blinker {{ 50% {{ opacity: 0; }} }}
        </style>
    </head>
    <body>
        <div class="container">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 2rem;">🧠</span>
                <h1>{selected_plant} AI Operations Center</h1>
            </div>
            <div class="right-panel">
                <div style="color: #00FF00;"><span class="blink-green">●</span> LIVE DATA FEED</div>
                <div id="clock" class="clock">Connecting...</div>
                <div class="sub-text">{date_str_py} (Sarawak Time)</div>
            </div>
        </div>
        <script>
            function update() {{
                const now = new Date();
                const utc = now.getTime() + (now.getTimezoneOffset() * 60000);
                const sarawakTime = new Date(utc + (3600000 * 8));
                document.getElementById('clock').innerText = sarawakTime.toLocaleTimeString('en-US', {{ hour12: false }});
            }}
            setInterval(update, 1000); update();
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=120)

@st.fragment(run_every=60) 
def render_main_content():
    render_iframe_header(time.time())
    df_raw, curr_data = get_weather_data(plant_data['lat'], plant_data['lon'])

    if not df_raw.empty:
        # --- GRID COMMANDER ---
        st.markdown("### ⚡ SEB Grid Commander (Auto-Dispatch)")
        c_input, c_status = st.columns([1, 2])
        with c_input:
            target_grid_load = st.number_input("🎯 Total Grid Demand (MW)", value=1000, step=100)
        
        dispatch_df, grid_msg, total_gen = calculate_fleet_dispatch(target_grid_load, efficiency_slider)
        with c_status:
            if "⚠️" in grid_msg: st.warning(f"{grid_msg} Max Cap: {total_gen:.0f} MW")
            else: st.success(f"{grid_msg} Fleet outputting {total_gen:.0f} MW.")
            st.altair_chart(alt.Chart(dispatch_df).mark_bar().encode(x='Dispatch (MW)', y=alt.Y('Dam', sort='-x'), color=alt.Color('Dam', scale=alt.Scale(scheme='spectral'))).properties(height=150), use_container_width=True)

        with st.expander("🧠 Grid Commander Logic (How decisions are made)", expanded=False):
            st.markdown(f"""
            The **Grid Commander** uses a **Weighted Fleet Dispatch Algorithm** to ensure stability across the entire Sarawak Energy network.
            
            **1. Asset Weighting:**
            * Each dam is assigned a "Availability Score" based on its **Max Capacity** multiplied by its **Real-Time Health (%)**.
            * *Example:* If Batang Ai is at 80% health, its score drops, and the Grid Commander automatically shifts load to Bakun.
            
            **2. The Dispatch Formula:**
            $$ Dispatch_{{Dam}} = TotalDemand \\times \\frac{{(Capacity_{{Dam}} \\times Health_{{Dam}})}}{{\\sum_{{i=1}}^{{n}} (Capacity_i \\times Health_i)}} $$
            
            **3. Strategic Rules:**
            * **Baleh Exclusion:** The system detects "Construction" status and sets its weight to 0 automatically.
            * **Load Balancing:** This method prevents "over-sweating" a single turbine. If one dam is maintenance-constrained, others pick up the slack instantly.
            """)

        st.divider()

        # --- SINGLE SITE ---
        df_sim = run_ai_simulation(df_raw.copy(), brand_eff, efficiency_slider, panel_area_input, plant_data['capacity_mw'])
        now_naive = datetime.now(pytz.timezone('Asia/Kuching')).replace(tzinfo=None)
        try: curr = df_sim.iloc[df_sim.index.get_indexer([now_naive], method='nearest')[0]]
        except: curr = df_sim.iloc[0]

        st.markdown(f"### 🏭 {selected_plant} Operations")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Hydro Setpoint", f"{curr['Hydro_MW']:.0f} MW", delta="Target Load")
        c2.metric("Suggested Flow", f"{curr['Flow_m3s']:.1f} m³/s")
        c3.metric("Gate Opening", f"{curr['Gate_Open_%']:.1f} %")
        c4.metric("Solar Predicted", f"{curr['Solar_MW']:.2f} MW")

        with st.expander("🧠 Single Dam Decision Logic (Micro-Dispatch)", expanded=True):
            st.markdown(f"""
            This module controls the specific behavior of **{selected_plant}** based on local sensors.
            
            **1. Solar Integration (The "Duck Curve" Fix):**
            * **Observation:** Solar output is currently **{curr['Solar_MW']:.2f} MW**.
            * **Action:** The Hydro Setpoint is dynamically adjusted: $Hydro = Demand - Solar$.
            * **Logic:** If a cloud passes over (Solar drops), the Hydro Gate immediately opens wider to fill the gap.
            
            **2. Economic Optimization:**
            * **Current State:** {curr['Ctx_State']}
            * **Price Signal:** The AI detects if we are in a PEAK or OFF-PEAK pricing window.
            * **Command:** "{curr['Instruction']}" -> This maximizes revenue by saving water when prices are low and discharging heavily when prices are high.
            """)

        with st.expander("📊 Mechanical vs Electrical Analysis", expanded=True):
            col_logic, col_curve = st.columns([1, 1.5])
            col_logic.markdown(f"**AI Mapping:** To reach **{curr['Hydro_MW']:.0f} MW**, turbines require **{curr['Flow_m3s']:.1f} m³/s**. Due to non-linear hydraulics, gates are set to **{curr['Gate_Open_%']:.1f}%**.")
            gate_range = np.linspace(0, 100, 100)
            power_range = (gate_range/100)**1.5 * plant_data['capacity_mw'] 
            curve_df = pd.DataFrame({'Gate %': gate_range, 'Power (MW)': power_range})
            base_curve = alt.Chart(curve_df).mark_line(color='#00FFFF', size=3).encode(x='Gate %', y='Power (MW)')
            dot = alt.Chart(pd.DataFrame({'Gate %': [curr['Gate_Open_%']], 'Power (MW)': [curr['Hydro_MW']]})).mark_circle(color='red', size=150).encode(x='Gate %', y='Power (MW)')
            col_curve.altair_chart((base_curve + dot).properties(height=250), use_container_width=True)

        st.divider()

# --- YIELD OPTIMIZATION ---
        st.subheader("💡 Yield Optimization & Maintenance Protocol")
        col_metrics, col_advice = st.columns([1, 2])
        with col_metrics:
            recoverable = curr['Recoverable_MW'] if curr['Recoverable_MW'] > 0 else 0
            st.metric("System Health (PR)", f"{efficiency_slider}%", f"-{100-efficiency_slider}% Loss", delta_color="inverse", help="PR: Plant Performance Ratio")
            st.caption("Real-time Operating Efficiency")
            st.metric("Recoverable Power", f"{recoverable:.2f} MW", "Opportunity Gain", delta_color="inverse", help="Energy recoverable via maintenance")
            st.caption("Lost Energy due to Environmental Factors")
        with col_advice:
            st.write(f"**🛠️ AI Diagnostic Checklist:**")
            if efficiency_slider >= 85:
                st.success("✅ **STATUS: OPTIMAL.**")
                st.checkbox("Surface Wash: Auto-sprinklers engaged.")
                st.checkbox("Algae check: No mooring debris.")
            else:
                st.error("🚨 **STATUS: CRITICAL YIELD LOSS.**")
                st.checkbox("String Check: Perform IV-curve tracing on Inverter Cluster B.")
                st.checkbox("Drone Scan: Hotspot detection on central arrays.")
                st.checkbox("DC Integrity: Combiner box moisture inspection.")

        st.divider()

        # --- ENV ---
        st.subheader("🔍 Environmental Intelligence")
        view_mode = st.radio("Select Data View:", ["☀️ Solar Intensity", "🌧️ Rain Radar", "☁️ Cloud Coverage"], horizontal=True)

        col_map, col_chart = st.columns([1, 2])
        with col_map:
            st.write(f"**📍 Location Map ({view_mode})**")
            lat, lon = plant_data['lat'], plant_data['lon']
            
            if "Solar" in view_mode:
                solar_val = safe_val(curr_data.get("Solar"), 0.0)
                
                # --- NEW GREEN HEATMAP VISUALIZATION ---
                # Logic: Scale the color brightness based on intensity (0-1000 W/m2)
                # Max intensity ~ 1000 W/m2
                intensity_scale = min(solar_val / 1000, 1.0)
                
                # Color Interpolation: Dark Green (0, 100, 0) -> Bright Lime (50, 255, 50)
                r = int(0 + (50 * intensity_scale))
                g = int(100 + (155 * intensity_scale))
                b = int(0 + (50 * intensity_scale))
                
                layers = [
                    # Large Radius Scatterplot to mimic "Region" Heatmap
                    pdk.Layer(
                        'ScatterplotLayer',
                        data=pd.DataFrame({'lat': [lat], 'lon': [lon]}),
                        get_position='[lon, lat]',
                        get_radius=3000, # Large radius to look like a map zone
                        get_fill_color=[r, g, b, 180], # Dynamic Green
                        stroked=True,
                        get_line_color=[255, 255, 255],
                        line_width_min_pixels=2
                    ),
                    # Text Layer for On-Map Stats (Like the reference image)
                    pdk.Layer(
                        'TextLayer',
                        data=pd.DataFrame({
                            'lat': [lat], 'lon': [lon], 
                            'text': [f"Est. PV Output: {curr['Solar_MW']:.1f} MW\nIrradiance: {solar_val:.0f} W/m²"]
                        }),
                        get_position='[lon, lat]',
                        get_text='text',
                        get_color=[255, 255, 255],
                        get_size=16,
                        get_alignment_baseline="'bottom'",
                        get_text_anchor="'middle'"
                    )
                ]
                
                # Use Dark Style map to match reference
                st.pydeck_chart(pdk.Deck(
                    map_style="mapbox://styles/mapbox/dark-v10", 
                    initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=11, pitch=0), 
                    layers=layers
                ))
            
            elif "Rain" in view_mode:
                windy_url = f"https://embed.windy.com/embed2.html?lat={lat}&lon={lon}&detailLat={lat}&detailLon={lon}&width=650&height=450&zoom=10&level=surface&overlay=rain&product=ecmwf&menu=&message=&marker=true&calendar=now&pressure=&type=map&location=coordinates&detail=&metricWind=default&metricTemp=default&radarRange=-1"
                components.iframe(windy_url, height=450)
                st.caption("📡 Live Radar (Windy.com)")
            
            elif "Cloud" in view_mode:
                windy_url = f"https://embed.windy.com/embed2.html?lat={lat}&lon={lon}&detailLat={lat}&detailLon={lon}&width=650&height=450&zoom=10&level=surface&overlay=clouds&product=ecmwf&menu=&message=&marker=true&calendar=now&pressure=&type=map&location=coordinates&detail=&metricWind=default&metricTemp=default&radarRange=-1"
                components.iframe(windy_url, height=450)
                st.caption("☁️ Live Clouds (Windy.com)")

        with col_chart:
            c_txt, c_btn = st.columns([4, 1])
            with c_txt: st.write(f"**📈 {view_mode} Analysis**")
            with c_btn: st.button("🔄 Optimize", help="Reset View")

            if "Solar" in view_mode:
                val = safe_val(curr_data.get('Solar'))
                c = make_chart(df_sim, 'Solar (W/m²)', '#FFD700', 'Irradiance (W/m²)', current_real_val=val)
                st.caption(f"Live Irradiance: {val:.0f} W/m²")
            elif "Rain" in view_mode:
                val = safe_val(curr_data.get('RainProb')) 
                c = make_chart(df_sim, 'Rain Probability (%)', '#0000FF', 'Probability (%)', current_real_val=val)
                st.caption(f"Rain Probability: {val:.0f}%")
            elif "Cloud" in view_mode:
                val = safe_val(curr_data.get('Cloud'))
                c = make_chart(df_sim, 'Cloud Cover (%)', '#808080', 'Cloud Cover (%)', current_real_val=val)
                st.caption(f"Current Cloud Cover: {val:.0f}%")
            st.altair_chart(c, width="stretch")

        st.divider()

        # --- DISPATCH CURVE ---
        st.subheader("⚙️ Dispatch Curve (48 Hours)")
        
        chart_data_disp = df_sim.resample('30min').mean(numeric_only=True).reset_index().melt('time', value_vars=['Hydro_MW', 'Solar_MW', 'Demand_MW'])
        
        base_chart = alt.Chart(chart_data_disp).encode(x=alt.X('time:T', axis=alt.Axis(format='%d %b %H:%M')))
        
        area_chart = base_chart.transform_filter(
            alt.FieldOneOfPredicate(field='variable', oneOf=['Hydro_MW', 'Solar_MW'])
        ).mark_area(opacity=0.8).encode(
            y=alt.Y('value:Q', stack=True), 
            color=alt.Color('variable:N', scale=alt.Scale(domain=['Hydro_MW', 'Solar_MW'], range=['#4c78a8', '#f58518'])),
            tooltip=['time:T', 'variable:N', 'value:Q']
        )
        
        line_chart = base_chart.transform_filter(
            alt.FieldOneOfPredicate(field='variable', oneOf=['Demand_MW'])
        ).mark_line(color='red', strokeDash=[5, 5]).encode(
            y='value:Q',
            tooltip=['time:T', 'variable:N', 'value:Q']
        )

        st.altair_chart((area_chart + line_chart).interactive(), width="stretch")
        
        with st.expander("📚 Prediction Methodology (LSTM & Heuristics)", expanded=True):
            st.markdown("""
            **1. Solar_MW Prediction (Machine Learning)**
            The AI uses a **Long Short-Term Memory (LSTM)** neural network.             * **Inputs:** Irradiance ($W/m^2$), Cloud Cover (%), and Precipitation.
            * **Logic:** The model estimates the angle of incidence relative to Sarawak's coordinates to forecast clear-sky power, then applies a cloud-density penalty.

            **2. Demand_MW Prediction (Grid Heuristics)**
            Demand is estimated using a **Stochastic Double-Peak Profile**.
            * **Cycle:** Simulates the industrial rise at 11:00 AM and residential surge at 7:00 PM.
            * **Resilience:** The red line represents the target load hydro must track to ensure grid frequency remains at 50Hz.
            """)

    else:
        st.warning("Connecting to Satellite Feed...")

if __name__ == "__main__":
    render_main_content()