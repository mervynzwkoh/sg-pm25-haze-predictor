"""Client for Singapore government open APIs (data.gov.sg).

Fetches hourly PM2.5, 24-hr PSI, wind speed, and wind direction data.
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import requests
import pandas as pd
from pytz import timezone

from src.config import DATAGOV_API_BASE, SG_REGIONS

logger = logging.getLogger(__name__)
SGT = timezone("Asia/Singapore")


class DataGovClient:
    """Handles communication with data.gov.sg APIs."""

    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "HazePredictor/1.0"})

    def fetch_latest_pm25(self) -> Dict[str, Any]:
        """Fetch real-time PM2.5 one-hourly readings across Singapore regions."""
        url = f"{DATAGOV_API_BASE}/pm25"
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json().get("data", {})
        items = data.get("items", [])
        if not items:
            return {}
        latest_item = items[-1]
        readings = latest_item.get("readings", {}).get("pm25_one_hourly", {})
        return {
            "timestamp": latest_item.get("timestamp"),
            "readings": readings,
        }

    def fetch_latest_psi(self) -> Dict[str, Any]:
        """Fetch real-time PSI twenty-four-hourly readings across Singapore regions."""
        url = f"{DATAGOV_API_BASE}/psi"
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json().get("data", {})
        items = data.get("items", [])
        if not items:
            return {}
        latest_item = items[-1]
        readings = latest_item.get("readings", {}).get("psi_twenty_four_hourly", {})
        return {
            "timestamp": latest_item.get("timestamp"),
            "readings": readings,
        }

    def fetch_historical_pm25(self, date_str: str) -> List[Dict[str, Any]]:
        """Fetch hourly PM2.5 readings for a specific date (YYYY-MM-DD)."""
        url = f"{DATAGOV_API_BASE}/pm25?date={date_str}"
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        items = resp.json().get("data", {}).get("items", [])
        records = []
        for it in items:
            ts = it.get("timestamp")
            readings = it.get("readings", {}).get("pm25_one_hourly", {})
            if ts and readings:
                rec = {"timestamp": ts}
                for reg in SG_REGIONS.keys():
                    rec[f"pm25_{reg}"] = readings.get(reg)
                records.append(rec)
        return records

    def fetch_historical_psi(self, date_str: str) -> List[Dict[str, Any]]:
        """Fetch 24-hr PSI readings for a specific date (YYYY-MM-DD)."""
        url = f"{DATAGOV_API_BASE}/psi?date={date_str}"
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        items = resp.json().get("data", {}).get("items", [])
        records = []
        for it in items:
            ts = it.get("timestamp")
            readings = it.get("readings", {}).get("psi_twenty_four_hourly", {})
            if ts and readings:
                rec = {"timestamp": ts}
                for reg in SG_REGIONS.keys():
                    rec[f"psi_{reg}"] = readings.get(reg)
                records.append(rec)
        return records

    def fetch_latest_wind(self) -> Dict[str, Any]:
        """Fetch current surface wind speed and direction from data.gov.sg v1."""
        speed_url = "https://api.data.gov.sg/v1/environment/wind-speed"
        dir_url = "https://api.data.gov.sg/v1/environment/wind-direction"

        speed_val = None
        dir_val = None
        timestamp = None

        try:
            r_s = self.session.get(speed_url, timeout=self.timeout)
            if r_s.status_code == 200:
                s_items = r_s.json().get("items", [])
                if s_items:
                    timestamp = s_items[-1].get("timestamp")
                    readings = s_items[-1].get("readings", [])
                    vals = [x["value"] for x in readings if "value" in x and x["value"] is not None]
                    if vals:
                        speed_val = sum(vals) / len(vals)  # knots or m/s avg across SG
        except Exception as e:
            logger.warning(f"Error fetching wind speed from data.gov.sg: {e}")

        try:
            r_d = self.session.get(dir_url, timeout=self.timeout)
            if r_d.status_code == 200:
                d_items = r_d.json().get("items", [])
                if d_items:
                    readings = d_items[-1].get("readings", [])
                    vals = [x["value"] for x in readings if "value" in x and x["value"] is not None]
                    if vals:
                        dir_val = sum(vals) / len(vals)  # degrees
        except Exception as e:
            logger.warning(f"Error fetching wind direction from data.gov.sg: {e}")

        return {
            "timestamp": timestamp,
            "wind_speed": speed_val,
            "wind_direction": dir_val,
        }

    def fetch_historical_pm25_range(self, days: int = 14) -> pd.DataFrame:
        """Fetch PM2.5 readings across the past N days with polite pacing to avoid rate limits."""
        import time
        now = datetime.now(SGT)
        all_records = []
        for i in range(days):
            target_date = (now - timedelta(days=i)).strftime("%Y-%m-%d")
            # Up to 2 retries on 429
            for attempt in range(2):
                try:
                    records = self.fetch_historical_pm25(target_date)
                    all_records.extend(records)
                    time.sleep(0.3)  # Gentle spacing
                    break
                except requests.exceptions.HTTPError as err:
                    if err.response is not None and err.response.status_code == 429:
                        time.sleep(1.0 * (attempt + 1))
                    else:
                        logger.warning(f"Failed to fetch PM2.5 for {target_date}: {err}")
                        break
                except Exception as e:
                    logger.warning(f"Failed to fetch PM2.5 for {target_date}: {e}")
                    break

        if not all_records:
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        return df
