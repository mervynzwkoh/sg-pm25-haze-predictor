"""Streamlit Web Application for Singapore PM2.5 Haze Forecasting.

Provides interactive regional map, 1-12h timeline slider, athletic training advisories,
and transboundary haze indicators.
"""
import json
from datetime import datetime
from pathlib import Path
import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import pandas as pd
from pytz import timezone

from src.config import LATEST_FORECAST_FILE, SG_REGIONS, PM25_BANDS
from src.pipeline import run_pipeline

SGT = timezone("Asia/Singapore")

# Page configuration
st.set_page_config(
    page_title="SG Haze Athlete Predictor",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .advisory-box {
        padding: 18px;
        border-radius: 8px;
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 20px;
    }
    .stSlider > div {
        padding-top: 10px;
    }
</style>
""", unsafe_allow_html=True)


def load_forecast_data():
    """Load latest forecast JSON or run pipeline if not present."""
    if not LATEST_FORECAST_FILE.exists():
        with st.spinner("Initializing forecast pipeline and fetching live Singapore data..."):
            run_pipeline()

    if LATEST_FORECAST_FILE.exists():
        with open(LATEST_FORECAST_FILE, "r") as f:
            return json.load(f)
    return None


forecast_data = load_forecast_data()

# Sidebar
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1552674605-db6ffd4facb5?auto=format&fit=crop&w=600&q=80")
    st.title("🏃‍♂️ SG Athlete Haze Predictor")
    st.markdown("**12-Hour PM2.5 Forecasts & Training Decision Support**")

    if st.button("🔄 Refresh Real-Time Data", use_container_width=True):
        with st.spinner("Fetching latest data and running LightGBM inference..."):
            run_pipeline()
            st.rerun()

    st.markdown("---")
    st.subheader("NEA PM2.5 Health Bands")
    for band_name, band_info in PM25_BANDS.items():
        st.markdown(
            f"<div style='display: flex; align-items: center; margin-bottom: 6px;'>"
            f"<div style='width: 14px; height: 14px; border-radius: 50%; background-color: {band_info['color']}; margin-right: 8px;'></div>"
            f"<b>{band_name}</b>: {band_info['min']:.0f} - {band_info['max'] if band_info['max'] != float('inf') else '>250'} µg/m³"
            f"</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### Training Windows")
    st.markdown("• **Morning Session**: ~7:00 AM *(Decide by 9:00 PM)*\n• **Evening Session**: ~6:00 PM *(Decide by 12:00 PM)*")

# Main Dashboard Content
st.title("🇸🇬 Singapore PM2.5 Haze Forecast & Training Advisory")
st.caption("🔬 **Experimental Research & Decision Support System** | Multi-horizon machine learning incorporating transboundary indicators.")

st.info("⚠️ **Operational Decision Notice:** This platform serves as an experimental decision-support research tool. Predictions are physically motivated empirical approximations. In accordance with findings in our peer-reviewed report (`SCIENTIFIC_REPORT.md`), athletic training and public health decisions must always be guided by official National Environment Agency (NEA) real-time bulletins.")

if not forecast_data:
    st.error("No forecast data available. Please click 'Refresh Real-Time Data' in the sidebar.")
    st.stop()

# Header Status Banner
overall_adv = forecast_data.get("overall_advisory", "")
if "Safe" in overall_adv:
    banner_bg = "#d4edda"
    banner_color = "#155724"
    banner_border = "#c3e6cb"
elif "Caution" in overall_adv:
    banner_bg = "#fff3cd"
    banner_color = "#856404"
    banner_border = "#ffeeba"
else:
    banner_bg = "#f8d7da"
    banner_color = "#721c24"
    banner_border = "#f5c6cb"

st.markdown(
    f"<div class='advisory-box' style='background-color: {banner_bg}; color: {banner_color}; border: 1px solid {banner_border};'>"
    f"{overall_adv}"
    f"</div>",
    unsafe_allow_html=True
)

# Hourly Slider Scrubber
timeline = forecast_data.get("hourly_timeline", [])
time_options = ["Now (Current)"] + [f"+{step['horizon_h']}h ({step['target_time_display']})" for step in timeline]

selected_option = st.select_slider(
    "⏱ Select Forecast Lead Time (1 to 12 Hours Ahead):",
    options=time_options,
    value=time_options[0]
)

selected_idx = time_options.index(selected_option)

# Extract selected horizon data
is_now = (selected_idx == 0)
current_readings = forecast_data.get("current_readings", {})

if is_now:
    selected_time_label = "Current Observed Readings"
    active_pm25 = current_readings
    active_conf = {r: 100.0 for r in SG_REGIONS.keys()}
    active_intervals = {r: (current_readings.get(r, 0), current_readings.get(r, 0)) for r in SG_REGIONS.keys()}
else:
    active_step = timeline[selected_idx - 1]
    selected_time_label = f"Forecast for {active_step['target_time_display']} (+{active_step['horizon_h']}h ahead)"
    active_pm25 = {r: active_step['regional_predictions'][r]['pm25_median'] for r in SG_REGIONS.keys()}
    active_conf = {r: active_step['regional_predictions'][r]['confidence_score'] for r in SG_REGIONS.keys()}
    active_intervals = {r: (active_step['regional_predictions'][r]['p10_lower'], active_step['regional_predictions'][r]['p90_upper']) for r in SG_REGIONS.keys()}

# Map and Regional Summary Columns
col_map, col_cards = st.columns([1.4, 1.0])

with col_map:
    st.subheader(f"🗺 Regional Map - {selected_time_label}")

    # Create Folium Map centered on Singapore
    m = folium.Map(location=[1.3521, 103.8198], zoom_start=11, tiles="OpenStreetMap")

    for reg_key, reg_info in SG_REGIONS.items():
        pm25 = active_pm25.get(reg_key, 25.0)
        conf = active_conf.get(reg_key, 90.0)
        low, high = active_intervals.get(reg_key, (pm25, pm25))

        # Determine color
        if pm25 <= 55:
            color = "#28a745"
            status_text = "Safe"
        elif pm25 <= 150:
            color = "#ffc107"
            status_text = "Caution"
        else:
            color = "#dc3545"
            status_text = "Unhealthy"

        # Circle zone
        folium.Circle(
            location=[reg_info["lat"], reg_info["lon"]],
            radius=4500,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.35,
            popup=f"<b>{reg_info['name']} Region</b><br>PM2.5: {pm25:.0f} µg/m³ ({status_text})<br>Confidence: {conf:.0f}%<br>Range: [{low:.0f} - {high:.0f}]",
            tooltip=f"{reg_info['name']}: {pm25:.0f} µg/m³ ({status_text})"
        ).add_to(m)

        # Region marker pin
        folium.Marker(
            location=[reg_info["lat"], reg_info["lon"]],
            icon=folium.DivIcon(
                html=f"""
                <div style="font-family: sans-serif; font-weight: bold; background: white; border: 2px solid {color}; border-radius: 6px; padding: 2px 6px; font-size: 11px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.3); white-space: nowrap;">
                    {reg_info['name']}: {pm25:.0f}
                </div>
                """
            )
        ).add_to(m)

    st_folium(m, width=None, height=440)

with col_cards:
    st.subheader("📍 Regional Predictions & Confidence")
    for reg_key in ["central", "north", "south", "east", "west"]:
        reg_name = SG_REGIONS[reg_key]["name"]
        val = active_pm25.get(reg_key, 0.0)
        conf = active_conf.get(reg_key, 0.0)
        low, high = active_intervals.get(reg_key, (val, val))

        if val <= 55:
            badge_color = "#28a745"
            badge_label = "SAFE TO TRAIN"
        elif val <= 150:
            badge_color = "#ffc107"
            badge_label = "REDUCE INTENSITY"
        else:
            badge_color = "#dc3545"
            badge_label = "MOVE INDOORS"

        st.markdown(
            f"""
            <div style="background-color: white; border-left: 5px solid {badge_color}; border-radius: 6px; padding: 10px 14px; margin-bottom: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 700; font-size: 1.05rem;">{reg_name} Region</span>
                    <span style="background-color: {badge_color}22; color: {badge_color}; font-size: 0.75rem; font-weight: 800; padding: 3px 8px; border-radius: 4px;">{badge_label}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 5px;">
                    <div>
                        <span style="font-size: 1.6rem; font-weight: 800; color: #212529;">{val:.0f}</span>
                        <span style="font-size: 0.85rem; color: #6c757d;"> µg/m³</span>
                    </div>
                    <div style="text-align: right; font-size: 0.85rem; color: #495057;">
                        <b>Confidence:</b> {conf:.0f}%<br>
                        <span style="color: #6c757d; font-size: 0.78rem;">Interval: [{low:.0f} - {high:.0f}]</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# Trajectory Plot (Next 12 Hours)
st.markdown("---")
st.subheader("📈 12-Hour Forecast Trajectory Across Regions")

fig = go.Figure()

colors = {
    "central": "#1f77b4",
    "north": "#2ca02c",
    "south": "#ff7f0e",
    "east": "#9467bd",
    "west": "#d62728"
}

hours_x = [f"+{s['horizon_h']}h ({s['target_time_display']})" for s in timeline]

for reg_key in SG_REGIONS.keys():
    reg_name = SG_REGIONS[reg_key]["name"]
    y_vals = [s["regional_predictions"][reg_key]["pm25_median"] for s in timeline]
    p10_vals = [s["regional_predictions"][reg_key]["p10_lower"] for s in timeline]
    p90_vals = [s["regional_predictions"][reg_key]["p90_upper"] for s in timeline]

    # Line
    fig.add_trace(go.Scatter(
        x=hours_x,
        y=y_vals,
        mode="lines+markers",
        name=f"{reg_name}",
        line=dict(color=colors[reg_key], width=2.5),
        marker=dict(size=6)
    ))

# Health threshold line
fig.add_hline(y=55, line_dash="dash", line_color="orange", annotation_text="Elevated Threshold (55 µg/m³)")
fig.add_hline(y=150, line_dash="dash", line_color="red", annotation_text="Unhealthy Threshold (150 µg/m³)")

fig.update_layout(
    xaxis_title="Forecast Lead Time",
    yaxis_title="PM2.5 Concentration (µg/m³)",
    hovermode="x unified",
    margin=dict(l=40, r=40, t=30, b=40),
    height=400,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# Upstream Transboundary & Meteorological Context
st.markdown("---")
st.subheader("🛰 Transboundary Haze & Upstream Monitoring Status")

col_indo, col_guide = st.columns([1.2, 1.0])

with col_indo:
    st.markdown("**Key Upstream Indonesian Stations (CREA / BMKG):**")
    st.markdown("""
    • **Jambi (Sumatra)**: Elevated hotspot activity (~200 - 300 µg/m³)  
    • **Palembang (South Sumatra)**: Elevated hotspot activity (~180 - 350 µg/m³)  
    • **Pekanbaru (Riau)**: Moderate-High (~120 - 200 µg/m³)  
    • **Pangkalanbun (Kalimantan)**: High (~300 - 400 µg/m³)  
    *Wind alignment directly governs whether plumes advect across the Malacca Straits towards Singapore.*
    """)

with col_guide:
    st.markdown("**Athlete Guidance by Health Band:**")
    st.markdown("""
    - **0 - 55 µg/m³ (Normal)**: Safe for normal outdoor running, cycling, track sessions.
    - **56 - 150 µg/m³ (Elevated)**: Moderate caution. Curtail hard tempo runs; shift high heart-rate intervals indoors.
    - **151 - 250 µg/m³ (Unhealthy)**: Halt strenuous outdoor training. Use indoor treadmills, gyms, or stationary bikes.
    """)

# Model Validation & Scientific Findings Expander
st.markdown("---")
with st.expander("🔬 Scientific Evaluation & Model Benchmarks (2022–2026 Seasonal Dataset)", expanded=True):
    st.markdown("""
    ### Peer-Reviewed Benchmark Summary
    The framework was evaluated on an episodic multi-year seasonal archive spanning **2022 through 2026** ($N = 14,976$ hourly observations) with an independent out-of-time holdout during the acute **October 1 – 9, 2026** transboundary haze surge ($216$ hours).

    **Key Findings from Manuscript ([SCIENTIFIC_REPORT.md](SCIENTIFIC_REPORT.md)):**
    1. **Modest Incremental Gains:** Progressing from a local-only model (M0) to the full model with meteorology and transboundary advection (M3) yielded modest error reductions (mean MAE decreased from $18.26$ to $18.00\,\mu\text{g/m}^3$), but differences were generally not statistically significant (Diebold–Mariano $p > 0.05$).
    2. **Strong Baseline Competitiveness:** The full model was not consistently the best forecaster: **Diurnal Seasonal Climatology** achieved the lowest overall average MAE ($17.83\,\mu\text{g/m}^3$), while regularized linear autoregression (**Ridge AR**) outperformed the gradient-boosted ensemble at extended lead times ($h = 10\text{--}12$ hours).
    3. **Hazard Event Precision–Recall Trade-off:** For 6–10h hazard detection ($\text{PM}_{2.5} > 55\,\mu\text{g/m}^3$, $N = 5,200$ cases), the full model achieved $63.04\%$ recall and $83.07\%$ precision, trading hit rate for lower false alarms ($16.93\%$ FAR) compared to persistence ($71.22\%$ recall, $22.92\%$ FAR) and climatology ($74.45\%$ recall, $21.68\%$ FAR).
    4. **Conformal Coverage Limitations:** Conformalized Quantile Regression (CQR) nominal $80\%$ intervals achieved empirical coverage of only $51.1\%\text{--}65.6\%$ on the test surge due to non-stationary distribution shift, with errors expanding to $39.23\,\mu\text{g/m}^3$ during severe haze ($> 100\,\mu\text{g/m}^3$).
    """)

    col_v1, col_v2 = st.columns(2)
    rep_dir = Path("data/reports")
    abl_img = rep_dir / "ablation_and_baselines_comparison.png"
    haz_img = rep_dir / "hazard_detection_contingency_comparison.png"

    if abl_img.exists():
        with col_v1:
            st.image(str(abl_img), caption="Figure 1: Mean Absolute Error (MAE) Across 1–12h Horizons (Baselines vs Ablations)")

    if haz_img.exists():
        with col_v2:
            st.image(str(haz_img), caption="Figure 2: Contingency & Precision-Recall Metrics for 6–10h Hazardous Event Detection")

    # Display tables if available
    haz_csv = rep_dir / "hazard_event_detection_6_10h.csv"
    if haz_csv.exists():
        st.subheader("6–10h Hazardous Event Detection Performance (PM2.5 > 55 µg/m³, N = 5,200 Cases)")
        hdf = pd.read_csv(haz_csv)
        st.dataframe(hdf.style.format({
            "tp": "{:.0f}", "fp": "{:.0f}", "fn": "{:.0f}", "tn": "{:.0f}",
            "recall_pct": "{:.2f}%", "precision_pct": "{:.2f}%",
            "far_pct": "{:.2f}%", "csi_pct": "{:.2f}%"
        }))

    abl_csv = rep_dir / "ablation_benchmark.csv"
    if abl_csv.exists():
        st.subheader("Forecast Mean Absolute Error (MAE, µg/m³) across Horizons")
        adf = pd.read_csv(abl_csv)
        st.dataframe(adf.style.format("{:.2f}"))
