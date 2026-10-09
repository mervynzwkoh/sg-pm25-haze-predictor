"""Unit tests for data ingestion clients and fallbacks."""
import pytest
from pathlib import Path
from src.data.indonesia_client import IndonesiaHazeData
from src.data.google_aqi_client import GoogleAirQualityClient
from src.data.openmeteo_client import OpenMeteoClient


def test_indonesia_haze_data_parsing():
    client = IndonesiaHazeData()
    assert not client.df.empty
    summary = client.get_regional_summary()
    assert "sumatra_mean_d1" in summary
    assert "kalimantan_mean_d1" in summary
    assert summary["station_count"] > 0
    assert summary["sumatra_mean_d1"] > 0

    readings = client.get_station_readings()
    assert "Batam" in readings
    assert "Pekanbaru" in readings
    assert readings["Batam"] > 0


def test_google_aqi_graceful_fallback():
    # Without an API key, client should report not available and return empty dict without failing
    client = GoogleAirQualityClient(api_key="")
    assert not client.is_available
    res = client.fetch_current_conditions()
    assert res == {}


def test_openmeteo_upstream_client():
    client = OpenMeteoClient()
    df = client.fetch_indonesia_upstream_aqi("Batam", past_days=1, forecast_days=1)
    assert not df.empty
    assert "timestamp" in df.columns
    assert "indo_pm25_batam" in df.columns
