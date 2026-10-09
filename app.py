"""Streamlit Web Application for Singapore PM2.5 Haze Forecasting.

Provides interactive regional map, 1-12h timeline slider, athletic training advisories,
and transboundary haze indicators.
"""
import json
from datetime import datetime
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
import folium
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
        padding: 14px 18px;
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 14px;
    }
</style>
""", unsafe_allow_html=True)


def load_forecast_data(force_refresh: bool = False):
    """Load latest forecast JSON or run fast pipeline if not present or stale."""
    now = datetime.now(SGT)
    need_refresh = force_refresh

    if not need_refresh and LATEST_FORECAST_FILE.exists():
        try:
            with open(LATEST_FORECAST_FILE, "r") as f:
                data = json.load(f)
            gen_time = datetime.fromisoformat(data["generated_at"])
            # Auto-refresh if data is older than 45 minutes or from an earlier hour
            if (now - gen_time).total_seconds() > 2700:
                need_refresh = True
            else:
                return data
        except Exception:
            need_refresh = True
    else:
        need_refresh = True

    if need_refresh:
        with st.spinner("Fetching latest live Singapore readings and running fast forecast..."):
            try:
                data = run_pipeline(force_retrain=False)
                return data
            except Exception as e:
                if LATEST_FORECAST_FILE.exists():
                    with open(LATEST_FORECAST_FILE, "r") as f:
                        return json.load(f)
                st.error(f"Error updating real-time forecast: {e}")
                return None


forecast_data = load_forecast_data()

# Sidebar
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1552674605-db6ffd4facb5?auto=format&fit=crop&w=600&q=80")
    st.title("🏃‍♂️ SG Athlete Haze Predictor")
    st.markdown("**12-Hour PM2.5 Forecasts & Training Decision Support**")

    if st.button("🔄 Refresh Real-Time Data", use_container_width=True):
        with st.spinner("Fetching latest live data and generating fast forecast..."):
            run_pipeline(force_retrain=False)
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
    st.markdown("• **Morning Session**: ~7:00 AM *(Alert at 10:00 PM SGT)*\n• **Evening Session**: ~6:00 PM *(Alert at 2:00 PM SGT)*")

# Main Dashboard Content
st.title("🇸🇬 Singapore PM2.5 Haze Forecast & Training Advisory")
st.caption("🔬 **Experimental Research & Decision Support System** | Multi-horizon machine learning incorporating transboundary indicators.")

with st.expander("⚠️ Operational Decision Notice & Model Advisory Guidelines", expanded=False):
    st.info("This platform serves as an experimental decision-support research tool. Predictions are physically motivated empirical approximations. Athletic training and public health decisions must always be guided by official National Environment Agency (NEA) real-time bulletins.")

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

# Extract timeline
timeline = forecast_data.get("hourly_timeline", [])

# Interactive Map, Real-time Scrubber & Regional Predictions
def render_interactive_dashboard(forecast_data):
    current_readings = forecast_data.get("current_readings", {})
    base_time = forecast_data.get("base_time_display", "")
    if not base_time:
        try:
            gen_dt = datetime.fromisoformat(forecast_data.get("generated_at", "")).astimezone(SGT)
            base_time = gen_dt.strftime("%I:%M %p")
        except Exception:
            base_time = datetime.now(SGT).strftime("%I:%M %p")

    regions_config = {
        "central": {"name": "Central", "lat": 1.357, "lon": 103.826},
        "north": {"name": "North", "lat": 1.418, "lon": 103.82},
        "south": {"name": "South", "lat": 1.295, "lon": 103.82},
        "east": {"name": "East", "lat": 1.357, "lon": 103.94},
        "west": {"name": "West", "lat": 1.357, "lon": 103.7}
    }

    steps = [{
        "horizon_h": 0,
        "displayLabel": f"Now (Current Observed - {base_time})",
        "headerLabel": f"Current Observed Readings ({base_time})",
        "regions": {
            r: {
                "pm25": float(current_readings.get(r, 25.0)),
                "conf": 100.0,
                "low": float(current_readings.get(r, 25.0)),
                "high": float(current_readings.get(r, 25.0))
            } for r in regions_config
        }
    }]

    for s in timeline:
        h = s["horizon_h"]
        td = s["target_time_display"]
        steps.append({
            "horizon_h": h,
            "displayLabel": f"+{h}h ({td})",
            "headerLabel": f"Forecast for {td} (+{h}h ahead)",
            "regions": {
                r: {
                    "pm25": float(s["regional_predictions"][r]["pm25_median"]),
                    "conf": float(s["regional_predictions"][r]["confidence_score"]),
                    "low": float(s["regional_predictions"][r]["p10_lower"]),
                    "high": float(s["regional_predictions"][r]["p90_upper"])
                } for r in regions_config
            }
        })

    steps_json = json.dumps(steps)
    regs_json = json.dumps(regions_config)

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8" />
      <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
      <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
      <style>
        * {{
          box-sizing: border-box;
          margin: 0;
          padding: 0;
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }}
        body {{
          background: transparent;
          color: #f1f5f9;
          padding: 4px;
          overflow: hidden;
        }}
        /* Scrubber Container */
        .scrubber-card {{
          background: #141822;
          border: 1px solid #283244;
          border-radius: 8px;
          padding: 10px 14px 8px 14px;
          margin-bottom: 12px;
        }}
        .scrubber-header {{
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 8px;
        }}
        .scrubber-title {{
          font-size: 0.92rem;
          font-weight: 700;
          color: #e2e8f0;
          display: flex;
          align-items: center;
          gap: 6px;
        }}
        .active-badge {{
          background: rgba(14, 165, 233, 0.15);
          color: #38bdf8;
          border: 1px solid rgba(14, 165, 233, 0.4);
          padding: 3px 10px;
          border-radius: 12px;
          font-size: 0.82rem;
          font-weight: 700;
          letter-spacing: 0.3px;
        }}
        .slider-input {{
          -webkit-appearance: none;
          appearance: none;
          width: 100%;
          height: 6px;
          border-radius: 3px;
          background: #2a3447;
          outline: none;
          cursor: pointer;
          margin-bottom: 6px;
        }}
        .slider-input::-webkit-slider-thumb {{
          -webkit-appearance: none;
          appearance: none;
          width: 18px;
          height: 18px;
          border-radius: 50%;
          background: #ef4444;
          cursor: pointer;
          box-shadow: 0 0 8px rgba(239, 68, 68, 0.7);
          border: 2px solid #ffffff;
          transition: transform 0.1s ease;
        }}
        .slider-input::-webkit-slider-thumb:hover {{
          transform: scale(1.15);
        }}
        .slider-input::-moz-range-thumb {{
          width: 18px;
          height: 18px;
          border-radius: 50%;
          background: #ef4444;
          cursor: pointer;
          box-shadow: 0 0 8px rgba(239, 68, 68, 0.7);
          border: 2px solid #ffffff;
        }}
        .ticks-bar {{
          display: flex;
          justify-content: space-between;
          font-size: 0.72rem;
          color: #64748b;
          padding: 0 2px;
        }}
        .ticks-bar span.active-tick {{
          color: #ef4444;
          font-weight: 800;
        }}

        /* Main Viewport Side-by-Side */
        .dashboard-row {{
          display: grid;
          grid-template-columns: 1.35fr 1fr;
          gap: 12px;
          height: 385px;
        }}
        .map-section {{
          display: flex;
          flex-direction: column;
          height: 100%;
        }}
        .panel-heading {{
          font-size: 0.95rem;
          font-weight: 700;
          color: #f8fafc;
          margin-bottom: 6px;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }}
        #map {{
          flex: 1;
          border-radius: 8px;
          border: 1px solid #283244;
          overflow: hidden;
        }}

        .cards-section {{
          display: flex;
          flex-direction: column;
          height: 100%;
        }}
        .cards-box {{
          flex: 1;
          display: flex;
          flex-direction: column;
          justify-content: space-between;
          gap: 5px;
          overflow-y: auto;
        }}
        .region-card {{
          background: #181d28;
          border: 1px solid #283347;
          border-left: 5px solid #28a745;
          border-radius: 6px;
          padding: 6px 12px;
          display: flex;
          flex-direction: column;
          justify-content: space-between;
          height: 60px;
          box-sizing: border-box;
          transition: border-color 0.15s ease, background-color 0.15s ease;
        }}
        .region-card:hover {{
          background: #1f2533;
          border-color: #3b475d;
        }}
        .card-row-1 {{
          display: flex;
          justify-content: space-between;
          align-items: center;
        }}
        .card-region-title {{
          font-weight: 700;
          font-size: 0.92rem;
          color: #ffffff !important;
        }}
        .card-badge {{
          font-size: 0.68rem;
          font-weight: 800;
          padding: 2px 7px;
          border-radius: 4px;
          letter-spacing: 0.4px;
          text-transform: uppercase;
        }}
        .card-row-2 {{
          display: flex;
          justify-content: space-between;
          align-items: baseline;
        }}
        .card-pm-num {{
          font-size: 1.42rem;
          font-weight: 800;
          color: #ffffff !important;
          line-height: 1;
        }}
        .card-pm-unit {{
          font-size: 0.76rem;
          color: #94a3b8 !important;
          margin-left: 3px;
          font-weight: 500;
        }}
        .card-meta-box {{
          text-align: right;
          line-height: 1.25;
        }}
        .card-confidence {{
          font-size: 0.76rem;
          color: #f1f5f9 !important;
          font-weight: 600;
        }}
        .card-interval {{
          font-size: 0.71rem;
          color: #94a3b8 !important;
        }}
      </style>
    </head>
    <body>
      <div class="scrubber-card">
        <div class="scrubber-header">
          <span class="scrubber-title">⏱ Forecast Lead Time Scrubber (1 to 12 Hours Ahead):</span>
          <span class="active-badge" id="scrubber-badge">Now</span>
        </div>
        <input type="range" id="scrubber-slider" min="0" max="12" value="0" step="1" class="slider-input" />
        <div class="ticks-bar" id="ticks-container">
          <span class="active-tick">Now</span>
          <span>+1h</span>
          <span>+2h</span>
          <span>+3h</span>
          <span>+4h</span>
          <span>+5h</span>
          <span>+6h</span>
          <span>+7h</span>
          <span>+8h</span>
          <span>+9h</span>
          <span>+10h</span>
          <span>+11h</span>
          <span>+12h</span>
        </div>
      </div>

      <div class="dashboard-row">
        <div class="map-section">
          <div class="panel-heading" id="map-heading">🗺 Regional Map - Current Observed Readings</div>
          <div id="map"></div>
        </div>

        <div class="cards-section">
          <div class="panel-heading">📍 Regional Predictions & Confidence</div>
          <div class="cards-box" id="cards-container">
            <!-- Region Cards populated via JS -->
          </div>
        </div>
      </div>

      <script>
        const steps = {steps_json};
        const regionsConfig = {regs_json};
        const regionOrder = ["central", "north", "south", "east", "west"];

        function getBand(pm25) {{
          if (pm25 <= 55) {{
            return {{
              color: "#28a745",
              badge: "SAFE TO TRAIN",
              badgeBg: "rgba(40, 167, 69, 0.18)",
              status: "Safe"
            }};
          }} else if (pm25 <= 150) {{
            return {{
              color: "#ffc107",
              badge: "REDUCE INTENSITY",
              badgeBg: "rgba(255, 193, 7, 0.18)",
              status: "Caution"
            }};
          }} else {{
            return {{
              color: "#dc3545",
              badge: "MOVE INDOORS",
              badgeBg: "rgba(220, 53, 69, 0.18)",
              status: "Unhealthy"
            }};
          }}
        }}

        // Initialize Cards DOM
        const cardsContainer = document.getElementById("cards-container");
        regionOrder.forEach(reg => {{
          const info = regionsConfig[reg];
          const card = document.createElement("div");
          card.className = "region-card";
          card.id = "card-" + reg;
          card.innerHTML = `
            <div class="card-row-1">
              <span class="card-region-title">${{info.name}} Region</span>
              <span class="card-badge" id="badge-${{reg}}">-</span>
            </div>
            <div class="card-row-2">
              <div>
                <span class="card-pm-num" id="val-${{reg}}">--</span>
                <span class="card-pm-unit">µg/m³</span>
              </div>
              <div class="card-meta-box">
                <div class="card-confidence" id="conf-${{reg}}">Confidence: --%</div>
                <div class="card-interval" id="interval-${{reg}}">Interval: [-- - --]</div>
              </div>
            </div>
          `;
          cardsContainer.appendChild(card);
        }});

        // Initialize Leaflet Map
        const map = L.map("map", {{
          center: [1.3521, 103.8198],
          zoom: 11,
          zoomControl: true,
          attributionControl: false
        }});

        L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
          maxZoom: 18
        }}).addTo(map);

        const circles = {{}};
        const markers = {{}};

        function createMarkerIcon(regKey, pm25, color) {{
          const name = regionsConfig[regKey].name;
          const html = `<div style="font-family: -apple-system, sans-serif; font-weight: 700; background: #ffffff; color: #0f172a; border: 2px solid ${{color}}; border-radius: 5px; padding: 2px 6px; font-size: 11px; text-align: center; box-shadow: 0 1px 4px rgba(0,0,0,0.35); white-space: nowrap;">${{name}}: ${{Math.round(pm25)}}</div>`;
          return L.divIcon({{ html: html, className: '', iconSize: [70, 22], iconAnchor: [35, 11] }});
        }}

        regionOrder.forEach(reg => {{
          const info = regionsConfig[reg];
          circles[reg] = L.circle([info.lat, info.lon], {{
            radius: 4300,
            color: "#28a745",
            fillColor: "#28a745",
            fillOpacity: 0.35,
            weight: 2
          }}).addTo(map);

          markers[reg] = L.marker([info.lat, info.lon], {{
            icon: createMarkerIcon(reg, 25, "#28a745")
          }}).addTo(map);
        }});

        function updateView(idx) {{
          const step = steps[idx];
          if (!step) return;

          document.getElementById("scrubber-badge").textContent = step.displayLabel;
          document.getElementById("map-heading").textContent = "🗺 Regional Map - " + step.headerLabel;

          const ticks = document.querySelectorAll("#ticks-container span");
          ticks.forEach((tick, i) => {{
            tick.classList.toggle("active-tick", i === idx);
          }});

          regionOrder.forEach(reg => {{
            const regData = step.regions[reg];
            const pm25 = regData.pm25;
            const conf = regData.conf;
            const low = regData.low;
            const high = regData.high;
            const band = getBand(pm25);

            if (circles[reg]) {{
              circles[reg].setStyle({{
                color: band.color,
                fillColor: band.color
              }});
              circles[reg].bindTooltip(
                `<b>${{regionsConfig[reg].name}} Region</b><br>PM2.5: ${{Math.round(pm25)}} µg/m³ (${{band.status}})<br>Confidence: ${{Math.round(conf)}}%<br>Range: [${{Math.round(low)}} - ${{Math.round(high)}}]`,
                {{ direction: 'top' }}
              );
            }}

            if (markers[reg]) {{
              markers[reg].setIcon(createMarkerIcon(reg, pm25, band.color));
            }}

            const card = document.getElementById("card-" + reg);
            if (card) {{
              card.style.borderLeftColor = band.color;
            }}
            const valEl = document.getElementById("val-" + reg);
            if (valEl) valEl.textContent = Math.round(pm25);

            const badgeEl = document.getElementById("badge-" + reg);
            if (badgeEl) {{
              badgeEl.textContent = band.badge;
              badgeEl.style.color = band.color;
              badgeEl.style.backgroundColor = band.badgeBg;
              badgeEl.style.border = "1px solid " + band.color + "55";
            }}

            const confEl = document.getElementById("conf-" + reg);
            if (confEl) confEl.textContent = "Confidence: " + Math.round(conf) + "%";

            const intervalEl = document.getElementById("interval-" + reg);
            if (intervalEl) intervalEl.textContent = "Interval: [" + Math.round(low) + " - " + Math.round(high) + "]";
          }});
        }}

        // Live input event for instantaneous slider scrubbing
        const slider = document.getElementById("scrubber-slider");
        slider.addEventListener("input", function(e) {{
          updateView(parseInt(e.target.value, 10));
        }});

        updateView(0);
        setTimeout(() => {{ map.invalidateSize(); }}, 200);
      </script>
    </body>
    </html>
    """

    components.html(html_code, height=485, scrolling=False)


render_interactive_dashboard(forecast_data)

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
    1. **Modest Incremental Gains:** Progressing from a local-only model (M0) to the full model with meteorology and transboundary advection (M3) yielded modest error reductions (mean MAE decreased from $18.26$ to $18.00\,\mu\text{g/m}^3$), but differences were generally not statistically significant (Diebold–Mariano $p > 0.05$ across horizons).
    2. **Strong Baseline Competitiveness:** The full model was not consistently the best forecaster: **Diurnal Seasonal Climatology** achieved the lowest overall average MAE ($17.83\,\mu\text{g/m}^3$), while regularized linear autoregression (**Ridge AR**) outperformed the gradient-boosted ensemble at extended lead times ($h = 10\text{--}12$ hours).
    3. **Conformal Coverage Limitations:** Conformalized Quantile Regression (CQR) nominal $80\%$ intervals achieved empirical coverage of only $51.1\%\text{--}65.6\%$ on the test surge due to non-stationary distribution shift, with errors expanding to $39.23\,\mu\text{g/m}^3$ during severe haze ($> 100\,\mu\text{g/m}^3$).
    """)

    rep_dir = Path("data/reports")

    # Publication Figures
    col_v1, col_v2 = st.columns(2)
    abl_img = rep_dir / "ablation_and_baselines_comparison.png"
    actual_img = rep_dir / "actual_vs_forecast_october_surge.png"
    if not actual_img.exists():
        actual_img = rep_dir / "models_vs_actual_october_surge.png"

    if abl_img.exists():
        with col_v1:
            st.image(str(abl_img), caption="Figure 1: Mean Absolute Error (MAE) Across 1–12h Horizons (Baselines vs Ablations)")

    if actual_img.exists():
        with col_v2:
            st.image(str(actual_img), caption="Figure 2: M3 Forecast vs Actual Ground Truth with 80% Uncertainty Band (October 2026 Surge)")

    # Interactive Plot: Model Predictions vs Actual Ground Truth
    surge_csv = rep_dir / "models_vs_actual_october_surge.csv"
    if surge_csv.exists():
        st.markdown("---")
        st.subheader("📉 Interactive Trajectory: Proposed M3 Model vs Actual Ground Truth")
        st.caption("Forecast tracking comparison between the proposed multi-horizon LightGBM model (M3, lead time h = 3 hours ahead) and actual ground truth observations during the acute October 1–9, 2026 haze surge, including the 80% uncertainty band [p10, p90].")

        sdf = pd.read_csv(surge_csv)
        sdf["timestamp"] = pd.to_datetime(sdf["timestamp"])

        fig_comp = go.Figure()

        # Actual Ground Truth
        fig_comp.add_trace(go.Scatter(
            x=sdf["timestamp"],
            y=sdf["actual_pm25"],
            mode="lines+markers",
            name="Actual Ground Truth (Central SG)",
            line=dict(color="#f8fafc", width=2.8),
            marker=dict(size=4)
        ))

        # Proposed LightGBM M3
        fig_comp.add_trace(go.Scatter(
            x=sdf["timestamp"],
            y=sdf["lgbm_m3_physics"],
            mode="lines",
            name="Proposed LightGBM M3 (+3h Ahead)",
            line=dict(color="#3b82f6", width=2.4)
        ))

        # 80% Uncertainty Band
        fig_comp.add_trace(go.Scatter(
            x=pd.concat([sdf["timestamp"], sdf["timestamp"][::-1]]),
            y=pd.concat([sdf["lgbm_p90"], sdf["lgbm_p10"][::-1]]),
            fill="toself",
            fillcolor="rgba(59, 130, 246, 0.20)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="M3 80% Uncertainty Band [p10, p90]"
        ))

        # Threshold lines
        fig_comp.add_hline(y=55, line_dash="dash", line_color="#f59e0b", annotation_text="Elevated (55 µg/m³)")
        fig_comp.add_hline(y=150, line_dash="dash", line_color="#ef4444", annotation_text="Unhealthy (150 µg/m³)")

        fig_comp.update_layout(
            xaxis_title="Date & Time (October 2026 SGT)",
            yaxis_title="PM2.5 Concentration (µg/m³)",
            hovermode="x unified",
            height=430,
            margin=dict(l=30, r=30, t=30, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    # Detailed Evaluation Results Benchmark Tables
    st.markdown("---")
    st.subheader("📊 Model Evaluation & Ablation Results across Forecast Horizons")

    tab_abl, tab_val = st.tabs(["Ablation & Baselines MAE Benchmark", "Multi-Horizon Accuracy & Coverage Metrics"])

    abl_csv = rep_dir / "ablation_benchmark.csv"
    if abl_csv.exists():
        with tab_abl:
            st.markdown("**Mean Absolute Error (MAE, µg/m³) & Relative Skill Gain across 1–12h Lead Times:**")
            adf = pd.read_csv(abl_csv)
            num_cols = adf.select_dtypes(include=["float", "float64"]).columns
            st.dataframe(adf.style.format({c: "{:.2f}" for c in num_cols}), use_container_width=True)

    val_csv = rep_dir / "validation_summary.csv"
    if val_csv.exists():
        with tab_val:
            st.markdown("**Comprehensive Validation Metrics (MAE, RMSE, MAPE, Skill Gain %, and 80% Prediction Interval Coverage):**")
            vdf = pd.read_csv(val_csv)
            v_num_cols = vdf.select_dtypes(include=["float", "float64"]).columns
            st.dataframe(vdf.style.format({c: "{:.2f}" for c in v_num_cols}), use_container_width=True)
