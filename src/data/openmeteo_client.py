"""Client for Open-Meteo Air Quality and Weather Forecast APIs.

Provides 100% free, keyless hourly meteorological and air quality data.
"""
import logging
from typing import Dict, Any, Optional
import requests
import pandas as pd

from src.config import OPEN_METEO_AQI_URL, OPEN_METEO_WEATHER_URL, SG_CENTER, INDONESIA_UPSTREAM_STATIONS

logger = logging.getLogger(__name__)


class OpenMeteoClient:
    """Fetches weather and atmospheric data from Open-Meteo."""

    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self.session = requests.Session()

    def fetch_singapore_meteo(self, past_days: int = 50, forecast_days: int = 2) -> pd.DataFrame:
        """Fetch hourly weather variables for Singapore including precipitation and wind vectors."""
        params = {
            "latitude": SG_CENTER["lat"],
            "longitude": SG_CENTER["lon"],
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m,surface_pressure",
            "past_days": past_days,
            "forecast_days": forecast_days,
            "timezone": "Asia/Singapore",
        }
        try:
            resp = self.session.get(OPEN_METEO_WEATHER_URL, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json().get("hourly", {})
            df = pd.DataFrame(data)
            if "time" in df.columns:
                df["timestamp"] = pd.to_datetime(df["time"]).dt.tz_localize("Asia/Singapore")
                df = df.drop(columns=["time"])
            return df
        except Exception as e:
            logger.error(f"Failed to fetch Open-Meteo weather for Singapore: {e}")
            return pd.DataFrame()

    def fetch_upstream_smoke_corridor(self, past_days: int = 50, forecast_days: int = 2) -> pd.DataFrame:
        """Fetch dynamic hourly smoke tracer data (PM2.5, CO, AOD) from upstream South Sumatra (Palembang)

        which physically advects towards Singapore.
        """
        params = {
            "latitude": -2.8986,  # Palembang, South Sumatra
            "longitude": 104.7014,
            "hourly": "pm2_5,carbon_monoxide,aerosol_optical_depth",
            "past_days": past_days,
            "forecast_days": forecast_days,
            "timezone": "Asia/Singapore",
        }
        try:
            resp = self.session.get(OPEN_METEO_AQI_URL, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json().get("hourly", {})
            df = pd.DataFrame(data)
            if "time" in df.columns:
                df["timestamp"] = pd.to_datetime(df["time"]).dt.tz_localize("Asia/Singapore")
                df = df.drop(columns=["time"])
            df = df.rename(columns={
                "pm2_5": "palembang_pm25",
                "carbon_monoxide": "palembang_co",
                "aerosol_optical_depth": "palembang_aod",
            })
            return df
        except Exception as e:
            logger.error(f"Failed to fetch upstream smoke corridor data: {e}")
            return pd.DataFrame()

    def fetch_singapore_air_quality(self, past_days: int = 14, forecast_days: int = 2) -> pd.DataFrame:
        """Fetch hourly air quality (PM2.5, PM10) reanalysis and model forecast for Singapore."""
        params = {
            "latitude": SG_CENTER["lat"],
            "longitude": SG_CENTER["lon"],
            "hourly": "pm2_5,pm10,carbon_monoxide,ozone",
            "past_days": past_days,
            "forecast_days": forecast_days,
            "timezone": "Asia/Singapore",
        }
        try:
            resp = self.session.get(OPEN_METEO_AQI_URL, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json().get("hourly", {})
            df = pd.DataFrame(data)
            if "time" in df.columns:
                df["timestamp"] = pd.to_datetime(df["time"]).dt.tz_localize("Asia/Singapore")
                df = df.drop(columns=["time"])
            # Prefix columns to avoid collisions
            df = df.rename(columns={
                "pm2_5": "openmeteo_pm25",
                "pm10": "openmeteo_pm10",
                "carbon_monoxide": "openmeteo_co",
                "ozone": "openmeteo_o3"
            })
            return df
        except Exception as e:
            logger.error(f"Failed to fetch Open-Meteo AQI for Singapore: {e}")
            return pd.DataFrame()

    def fetch_indonesia_upstream_aqi(self, station_name: str, past_days: int = 14, forecast_days: int = 2) -> pd.DataFrame:
        """Fetch hourly PM2.5 and wind from an Indonesian upstream hotspot coordinate."""
        if station_name not in INDONESIA_UPSTREAM_STATIONS:
            raise ValueError(f"Unknown station: {station_name}")

        info = INDONESIA_UPSTREAM_STATIONS[station_name]
        params = {
            "latitude": info["lat"],
            "longitude": info["lon"],
            "hourly": "pm2_5",
            "past_days": past_days,
            "forecast_days": forecast_days,
            "timezone": "Asia/Singapore",
        }
        try:
            resp = self.session.get(OPEN_METEO_AQI_URL, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json().get("hourly", {})
            df = pd.DataFrame(data)
            if "time" in df.columns:
                df["timestamp"] = pd.to_datetime(df["time"]).dt.tz_localize("Asia/Singapore")
                df = df.drop(columns=["time"])
            col_name = f"indo_pm25_{station_name.lower().replace(' ', '_')}"
            df = df.rename(columns={"pm2_5": col_name})
            return df
        except Exception as e:
            logger.error(f"Failed to fetch Open-Meteo upstream AQI for {station_name}: {e}")
            return pd.DataFrame()
