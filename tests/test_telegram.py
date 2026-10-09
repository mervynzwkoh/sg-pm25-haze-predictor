"""Unit tests for Telegram notification formatting and triggers."""
import pytest
from src.notifications.telegram_bot import TelegramNotifier


def test_telegram_message_formatting():
    notifier = TelegramNotifier()

    sample_forecast = {
        "overall_advisory": "✅ Safe to Train: All regions forecast below 55 µg/m³.",
        "hourly_timeline": [
            {
                "horizon_h": 6,
                "target_time_display": "06:00 PM",
                "sg_average": 38.0,
                "max_pm25": 44.0,
                "regional_predictions": {
                    "central": {"pm25_median": 35.0, "confidence_score": 88.0, "health_band": "Normal"},
                    "north": {"pm25_median": 32.0, "confidence_score": 85.0, "health_band": "Normal"},
                    "south": {"pm25_median": 40.0, "confidence_score": 82.0, "health_band": "Normal"},
                    "east": {"pm25_median": 30.0, "confidence_score": 90.0, "health_band": "Normal"},
                    "west": {"pm25_median": 44.0, "confidence_score": 80.0, "health_band": "Normal"},
                }
            }
        ]
    }

    msg = notifier.format_forecast_message(sample_forecast, trigger_type="scheduled")
    assert "SG HAZE BRIEFING" in msg
    assert "Central" in msg
    assert "North" in msg
    assert "South" in msg
    assert "Safe to Train" in msg
    assert "Conf: 88%" in msg


def test_telegram_unconfigured_safety():
    # When unconfigured, send_message should safely return False without crashing
    notifier = TelegramNotifier(token="", chat_id="")
    assert not notifier.is_configured
    assert notifier.send_message("Test message") is False
