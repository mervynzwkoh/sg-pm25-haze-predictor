# 🇸🇬 Singapore PM2.5 Haze Forecasting System

An experimental multi-horizon machine learning framework designed to predict hourly fine particulate matter ($\text{PM}_{2.5}$) concentrations across Singapore's five administrative planning regions (**North, South, East, West, Central**) up to **12 hours ahead**.

The system integrates local ground telemetry, surface meteorology, and physically motivated transboundary indicators from upstream Indonesian monitoring stations across Sumatra and Kalimantan. 

> [!IMPORTANT]
> **Operational Decision Support Disclaimer:**  
> This system is an **experimental research and decision-support tool**. While transboundary indicators improve hazardous-event recall over local autoregression, the models trade recall for lower false alarms compared to simpler baselines, and prediction intervals suffer coverage degradation during acute non-stationary surges. Athletic training protocols, school physical education classes, and public health advisories must always be guided by official real-time bulletins from Singapore's **National Environment Agency (NEA)**.
>
> 📄 **Full Scientific Manuscript:** For peer-review documentation of methodology, causal audits, and empirical benchmarks, see [SCIENTIFIC_REPORT.md](SCIENTIFIC_REPORT.md).

---

## 🔬 Core Scientific Findings

Evaluated on an episodic multi-year seasonal archive spanning **2022 through 2026** ($N = 14,976$ hourly observations) with an independent holdout during the acute **October 1–9, 2026** transboundary haze surge ($216$ hours):

1. **Modest Incremental Gains:** Adding meteorological and transboundary proxies reduced mean MAE from $18.26\,\mu\text{g/m}^3$ (local-only M0) to $18.00\,\mu\text{g/m}^3$ (full model M3). Improvements were consistent across horizons (+1.35% mean gain) but modest in magnitude ($0.26\,\mu\text{g/m}^3$) and generally **not statistically significant** (Diebold–Mariano $p > 0.05$).
2. **Strong Baseline Competitiveness:** The full model was not consistently the best forecaster:
   - **Diurnal Seasonal Climatology** achieved the lowest overall average MAE across the 12-hour horizon (**$17.83\,\mu\text{g/m}^3$** vs $18.00\,\mu\text{g/m}^3$ for M3).
   - Regularized linear autoregression (**Ridge AR**) outperformed the gradient-boosted ensemble at extended lead times ($h = 10\text{--}12$ hours, achieving $22.33\,\mu\text{g/m}^3$ at 10h vs $23.71\,\mu\text{g/m}^3$ for M3).
3. **Hazard Detection Precision–Recall Trade-offs:** For 6–10-hour hazardous event forecasting ($\text{PM}_{2.5} > 55\,\mu\text{g/m}^3$, $N = 5,200$ cases):
   - M3 improved recall over local-only M0 ($63.04\%$ vs $58.29\%$).
   - However, M3 achieved lower recall than persistence ($71.22\%$) and diurnal climatology ($74.45\%$), while achieving higher precision ($83.07\%$) and a lower false alarm ratio ($16.93\%$ vs $21.7\%\text{--}22.9\%$).
4. **Calibration Limits During Non-Stationary Surges:** Conformalized Quantile Regression (CQR) nominal 80% intervals achieved only $51.1\%\text{--}65.6\%$ empirical coverage on the test surge due to distribution shift. Error expanded to an MAE of $39.23\,\mu\text{g/m}^3$ during severe haze ($> 100\,\mu\text{g/m}^3$).

---

## 🏃‍♂️ System Capabilities

- **12-Hour Multi-Region Forecasting**: Direct multi-step quantile regression ($h \in [1..12]$ hours) across North, South, East, West, and Central Singapore.
- **Probabilistic Quantile Bounds**: Predicts 10th percentile ($p_{10}$), median ($p_{50}$), and 90th percentile ($p_{90}$) intervals using residual LightGBM ensembles.
- **Physically Motivated Transboundary Proxies**:
  - Upstream Indonesian monitoring telemetry (Sumatra: Jambi, Palembang, Pekanbaru; Kalimantan: Pangkalanbun) via CREA / BMKG.
  - Forward azimuth bearings and cosine directional wind alignment scores.
  - Multi-hour transit corridor lags ($\tau \in \{6, 12, 18\}$ h) and 3-hour precipitation washout proxies.
- **Athlete Health Bands & Decision Guidance**:
  - Maps forecasts to NEA $\text{PM}_{2.5}$ bands: **Normal** ($\le 55$), **Elevated** ($56\text{--}150$), **Unhealthy** ($151\text{--}250$), and **Hazardous** ($> 250\,\mu\text{g/m}^3$).
  - Actionable workout recommendations for morning (~07:00 SGT) and evening (~18:00 SGT) training sessions.
- **Interactive Streamlit Web Dashboard**:
  - Interactive map of Singapore with regional color-coded zones.
  - 1–12h timeline slider scrubber.
  - 12-hour trajectory charts and transparent scientific validation expander.
- **Automated Telegram Bot Alerts**:
  - Scheduled athlete briefings at **9:00 PM SGT** (morning workout prep) and **12:00 PM SGT** (evening workout prep).
  - Urgent threshold alerts when $\text{PM}_{2.5}$ is forecast to exceed $55\,\mu\text{g/m}^3$.
- **Free & Keyless Pipeline**:
  - Relies on open APIs (`data.gov.sg`, Open-Meteo, CREA ground station datasets).
  - Optional Google Air Quality API key support with seamless automatic fallback.

---

## 📐 Architecture & Pipeline Flow

```
[data.gov.sg (PM2.5, Meteo)]   [Open-Meteo (Reanalysis)]   [CREA Indonesia (Sumatra/Kalimantan)]
                                     │
                                     ▼
                      [Data Ingestion & Caching Engine]
                                     │
                                     ▼
                  [Causal Feature Engineering Pipeline]
                  • Autoregressive lags (t-1, 2, 3, 6, 12, 24)
                  • Backward rolling statistics (3h, 6h, 12h, 24h)
                  • Wind vector decomposition (u, v)
                  • Forward azimuth & cosine alignment proxies
                  • Precipitation scavenging proxies
                                     │
                                     ▼
                 [Multi-Horizon Quantile Residual Ensembles]
                 • Regional LightGBM models (h = 1 to 12)
                 • Conformalized Quantile Regression (CQR) calibration
                                     │
                                     ▼
                     [Advisory & Decision Support Engine]
                                     │
               ┌─────────────────────┴─────────────────────┐
               ▼                                           ▼
       [Streamlit Web App]                     [Telegram Bot Notifier]
       (Interactive Map & Slider)              (9pm / 12pm SGT & Spike Alerts)
```

---

## 📊 Benchmark Evaluation Summary

### 1. Forecast Accuracy Across Horizons (MAE, $\mu\text{g/m}^3$)
*Evaluated on the October 2026 holdout surge across Singapore regions.*

| Horizon $h$ | Persistence | Diurnal Climatology | Ridge AR | M0 (Local Only) | M1 (Local + Meteo) | M2 (+ Upstream) | M3 (Full Model) | M3 vs M0 Gain (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1h** | 7.11 | 7.12 | 7.49 | 7.09 | 7.06 | 7.05 | **7.04** | $+0.77$ |
| **3h** | 13.02 | 12.89 | 14.21 | 13.04 | **12.81** | 12.85 | 12.83 | $+1.55$ |
| **6h** | 18.19 | 17.93 | 19.51 | 18.20 | 17.94 | 17.91 | **17.84** | $+2.03$ |
| **8h** | 21.33 | **21.01** | 21.33 | 21.42 | 21.36 | 21.43 | 21.32 | $+0.53$ |
| **10h** | 23.76 | 23.26 | **22.33** | 23.86 | 23.81 | 23.79 | 23.71 | $+0.56$ |
| **12h** | 25.11 | 24.46 | **23.45** | 25.77 | 25.19 | 25.05 | 25.12 | $+2.41$ |
| **Mean** | **18.12** | **17.83** | **18.26** | **18.26** | **18.08** | **18.06** | **18.00** | **+1.35** |

### 2. 6–10-Hour Hazardous Event Detection ($\text{PM}_{2.5} > 55\,\mu\text{g/m}^3$, $N = 5,200$ Cases)

| Architecture | True Positives | False Positives | False Negatives | Recall (%) | Precision (%) | False Alarm Ratio (%) | Critical Success Index (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence Baseline** | 1,873 | 557 | 757 | **71.22** | 77.08 | 22.92 | **58.77** |
| **Diurnal Climatology** | 1,958 | 542 | 672 | **74.45** | 78.32 | 21.68 | **61.73** |
| **Ridge AR (Linear)** | 1,647 | 264 | 983 | 62.62 | **86.19** | **13.81** | 56.91 |
| **M0 (Local Only)** | 1,533 | 289 | 1,097 | 58.29 | 84.14 | 15.86 | 52.52 |
| **M3 (Full Model)** | 1,658 | 338 | 972 | 63.04 | 83.07 | 16.93 | 55.86 |

---

## 🚀 Quickstart & Local Setup

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/<your-username>/haze-predictor.git
cd haze-predictor

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables (Optional)
Create a `.env` file in the root directory:
```env
# Optional: Telegram Bot Alerts
TELEGRAM_BOT_TOKEN="your_bot_token_from_botfather"
TELEGRAM_CHAT_ID="your_telegram_group_or_chat_id"

# Optional: Google Air Quality API (leave empty to use free APIs only)
GOOGLE_AQI_API_KEY=""
```

### 3. Run Pipeline (Ingest, Train & Forecast)
```bash
python -m src.pipeline
```
This fetches real-time readings, runs inference, outputs metrics, and writes `data/latest_forecast.json`.

To test sending a Telegram alert immediately:
```bash
python -m src.pipeline --force-notify
```

### 4. Launch the Interactive Web Dashboard
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 5. Run Automated Tests
```bash
python -m pytest tests/
```

---

## 📱 Telegram Bot Setup (Free)

1. Open Telegram and search for `@BotFather`.
2. Send `/newbot`, follow prompts, and copy the **HTTP API Token**.
3. Create your training group chat, add your new bot, and send a message.
4. Get your chat ID by visiting `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` and locate `"chat":{"id": -123456789}`.
5. Set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in your `.env` or GitHub Secrets.

---

## 🌐 Free Public Deployment Guide

### Option A: Streamlit Community Cloud (Recommended)
1. Push your repository to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io/) with your GitHub account.
3. Click **New app**, select your repository, branch (`main`), and file path (`app.py`).
4. Click **Deploy!** Your app is live at a permanent public URL (e.g. `https://<your-app>.streamlit.app`).
5. (Optional) In app settings under **Secrets**, set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`.

### Option B: GitHub Actions Hourly Scheduled Job
The repository includes `.github/workflows/forecast.yml` which triggers automatically every hour (`cron: '0 * * * *'`):
1. In your GitHub repository, navigate to **Settings** > **Secrets and variables** > **Actions**.
2. Add Repository Secrets:
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
   - `GOOGLE_AQI_API_KEY` (optional)
3. Under **Settings** > **Actions** > **General**, ensure **Workflow permissions** is set to **Read and write permissions** (so the action can commit `latest_forecast.json`).

---

## 📄 License & Attribution
MIT License. Open for public and athlete community use. Ground observations courtesy of `data.gov.sg`, Open-Meteo, and CREA / BMKG.
