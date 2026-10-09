"""Configuration settings for Singapore PM2.5 haze forecasting system."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LATEST_FORECAST_FILE = DATA_DIR / "latest_forecast.json"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Singapore Planning Regions & Centroid Coordinates
SG_REGIONS = {
    "north": {"lat": 1.4180, "lon": 103.8200, "name": "North"},
    "south": {"lat": 1.2958, "lon": 103.8200, "name": "South"},
    "east": {"lat": 1.3570, "lon": 103.9400, "name": "East"},
    "west": {"lat": 1.3570, "lon": 103.7000, "name": "West"},
    "central": {"lat": 1.3570, "lon": 103.8200, "name": "Central"},
}

# Singapore Approximate Geographic Center
SG_CENTER = {"lat": 1.3521, "lon": 103.8198}

# Indonesia Key Upstream Monitoring / Fire Cluster Stations
# Distances and approximate bearings relative to Singapore center
INDONESIA_UPSTREAM_STATIONS = {
    "Batam": {"lat": 1.1301, "lon": 104.0529, "bearing_to_sg": 315.0, "distance_km": 35.0},
    "Pekanbaru": {"lat": 0.5071, "lon": 101.4478, "bearing_to_sg": 68.0, "distance_km": 280.0},
    "Kota Jambi": {"lat": -1.6101, "lon": 103.6131, "bearing_to_sg": 5.0, "distance_km": 330.0},
    "Muaro Jambi": {"lat": -1.5900, "lon": 103.7800, "bearing_to_sg": 2.0, "distance_km": 325.0},
    "Talang Betutu Palembang": {"lat": -2.8986, "lon": 104.7014, "bearing_to_sg": 345.0, "distance_km": 480.0},
    "Pangkalanbun": {"lat": -2.6842, "lon": 111.6253, "bearing_to_sg": 290.0, "distance_km": 980.0},
    "Kototabang": {"lat": -0.2019, "lon": 100.3208, "bearing_to_sg": 74.0, "distance_km": 420.0},
}

# Forecasting Horizon Settings
FORECAST_HORIZONS = list(range(1, 13))  # 1 to 12 hours ahead
QUANTILES = [0.10, 0.50, 0.90]

# NEA / MOH PM2.5 Health Bands (µg/m³) & Athletic Recommendations
PM25_BANDS = {
    "Normal": {
        "min": 0.0,
        "max": 55.0,
        "color": "#28a745",  # Green
        "badge": "Safe",
        "advisory": "Air quality is good/moderate. Safe for all outdoor athletic training sessions."
    },
    "Elevated": {
        "min": 55.1,
        "max": 150.0,
        "color": "#ffc107",  # Yellow/Orange
        "badge": "Caution",
        "advisory": "Elevated PM2.5 levels. Reduce strenuous outdoor exercise duration or intensity. Athletes with respiratory sensitivities should train indoors."
    },
    "Unhealthy": {
        "min": 150.1,
        "max": 250.0,
        "color": "#fd7e14",  # Orange/Red
        "badge": "Unhealthy",
        "advisory": "Unhealthy air quality. Avoid strenuous outdoor training. Shift training sessions indoors."
    },
    "Very Unhealthy": {
        "min": 250.1,
        "max": float("inf"),
        "color": "#dc3545",  # Red
        "badge": "Hazardous",
        "advisory": "Hazardous air quality. Halt all outdoor athletic workouts immediately."
    }
}

# API URLs & Endpoints
DATAGOV_API_BASE = "https://api-open.data.gov.sg/v2/real-time/api"
OPEN_METEO_AQI_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
OPEN_METEO_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
GOOGLE_AQI_API_URL = "https://airquality.googleapis.com/v1/currentConditions:lookup"

# Environment Variables
GOOGLE_AQI_API_KEY = os.getenv("GOOGLE_AQI_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Scheduled Notification Times (Singapore Time, UTC+8)
SCHEDULED_HOURS_SGT = [12, 21]  # 12:00 PM (for ~6pm training) and 9:00 PM (for ~7am morning training)
SPIKE_THRESHOLD_PM25 = 55.0     # Alert if PM2.5 spikes above normal within 3 hours
