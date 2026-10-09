"""Optional client for Google Air Quality API.

Provides graceful fallback when the API key is not configured or rate limits are reached.
"""
import logging
from typing import Dict, Any, Optional
import requests

from src.config import GOOGLE_AQI_API_URL, GOOGLE_AQI_API_KEY, SG_CENTER

logger = logging.getLogger(__name__)


class GoogleAirQualityClient:
    """Queries Google Air Quality REST API if credentials are provided."""

    def __init__(self, api_key: Optional[str] = None, timeout: int = 10):
        self.api_key = api_key or GOOGLE_AQI_API_KEY
        self.timeout = timeout

    @property
    def is_available(self) -> bool:
        """Check if an API key is configured."""
        return bool(self.api_key and self.api_key.strip())

    def fetch_current_conditions(self, lat: float = SG_CENTER["lat"], lon: float = SG_CENTER["lon"]) -> Dict[str, Any]:
        """Fetch current air quality conditions for given coordinates."""
        if not self.is_available:
            logger.info("Google AQI API key not set; skipping Google Air Quality request.")
            return {}

        url = f"{GOOGLE_AQI_API_URL}?key={self.api_key}"
        payload = {
            "location": {
                "latitude": lat,
                "longitude": lon
            },
            "extraComputations": [
                "POLLUTANT_ADDITIONAL_INFO",
                "DOMINANT_POLLUTANT_CONCENTRATION"
            ]
        }

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                # Extract PM2.5 pollutant concentration if present
                pm25_val = None
                pollutants = data.get("pollutants", [])
                for p in pollutants:
                    if p.get("code") == "pm25":
                        pm25_val = p.get("concentration", {}).get("value")
                        break
                return {
                    "source": "google_air_quality",
                    "dateTime": data.get("dateTime"),
                    "google_pm25": pm25_val,
                    "indexes": data.get("indexes", []),
                }
            elif resp.status_code in (401, 403):
                logger.warning(f"Google AQI API key invalid or unauthorized (HTTP {resp.status_code}). Falling back to free APIs.")
                return {}
            elif resp.status_code == 429:
                logger.warning("Google AQI API quota reached (HTTP 429). Falling back to free APIs.")
                return {}
            else:
                logger.warning(f"Google AQI request failed with HTTP {resp.status_code}: {resp.text}")
                return {}
        except Exception as e:
            logger.warning(f"Exception connecting to Google AQI API: {e}. Falling back to free APIs.")
            return {}
