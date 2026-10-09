"""Real-time inference engine for 12-hour multi-region PM2.5 forecasting.

Generates calibrated confidence scores, health risk bands, athletic training advisories,
and saves the output to latest_forecast.json.
"""
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from pytz import timezone

from src.config import (
    SG_REGIONS,
    FORECAST_HORIZONS,
    PM25_BANDS,
    LATEST_FORECAST_FILE,
)
from src.models.train import MultiHorizonForecaster

logger = logging.getLogger(__name__)
SGT = timezone("Asia/Singapore")


def compute_confidence_score(p10: float, p50: float, p90: float, horizon_h: int) -> float:
    """Compute a calibrated confidence percentage (0-100%) based on prediction interval spread

    and forecast horizon uncertainty decay.
    """
    spread = max(0.0, p90 - p10)
    baseline_val = max(p50, 15.0)
    relative_spread = spread / baseline_val

    # Horizon uncertainty decay factor
    horizon_factor = 1.0 - (0.025 * (horizon_h - 1))

    # Raw score based on interval sharpness
    raw_confidence = 100.0 * (1.0 - (0.45 * relative_spread)) * horizon_factor

    # Bound within realistic athletic confidence bounds [25%, 98%]
    calibrated = max(25.0, min(98.0, raw_confidence))
    return round(calibrated, 1)


def get_health_band(pm25_val: float) -> Dict[str, Any]:
    """Map a PM2.5 value to the corresponding NEA/MOH health category and advisory."""
    for band_name, band_info in PM25_BANDS.items():
        if band_info["min"] <= pm25_val <= band_info["max"]:
            return {
                "band": band_name,
                "badge": band_info["badge"],
                "color": band_info["color"],
                "advisory": band_info["advisory"],
            }
    return {
        "band": "Very Unhealthy",
        "badge": "Hazardous",
        "color": "#dc3545",
        "advisory": "Hazardous air quality. Halt outdoor athletic activity.",
    }


def generate_12h_forecast(
    forecaster: MultiHorizonForecaster,
    latest_feature_row: pd.DataFrame,
    current_readings: Dict[str, float],
    reference_time: Optional[datetime] = None
) -> Dict[str, Any]:
    """Generate 12-hour forecasts across all 5 Singapore regions."""
    ref_time = reference_time or datetime.now(SGT)

    forecast_output = {
        "generated_at": ref_time.isoformat(),
        "status": "success",
        "current_readings": current_readings,
        "regions": {},
        "hourly_timeline": [],
        "overall_advisory": "",
    }

    # Initialize region containers
    for reg, reg_info in SG_REGIONS.items():
        forecast_output["regions"][reg] = {
            "name": reg_info["name"],
            "lat": reg_info["lat"],
            "lon": reg_info["lon"],
            "current_pm25": current_readings.get(reg, 25.0),
            "forecasts": [],
        }

    base_time = ref_time.replace(minute=0, second=0, microsecond=0)
    forecast_output["base_time_display"] = base_time.strftime("%I:%M %p")

    # Generate predictions for each horizon h=1..12
    max_forecast_across_sg = 0.0

    for h in FORECAST_HORIZONS:
        forecast_timestamp = base_time + timedelta(hours=h)
        hour_summary = {
            "horizon_h": h,
            "target_time": forecast_timestamp.isoformat(),
            "target_time_display": forecast_timestamp.strftime("%I:%M %p"),
            "regional_predictions": {},
            "sg_average": 0.0,
            "max_pm25": 0.0,
        }

        sg_sum = 0.0
        h_max = 0.0

        for reg in SG_REGIONS.keys():
            cur_reg_val = current_readings.get(reg)
            preds = forecaster.predict_horizon(latest_feature_row, reg, h, current_val=cur_reg_val)
            conf = compute_confidence_score(preds["p10"], preds["p50"], preds["p90"], h)
            band = get_health_band(preds["p50"])

            reg_data = {
                "horizon_h": h,
                "target_time": forecast_timestamp.isoformat(),
                "time_display": forecast_timestamp.strftime("%I:%M %p"),
                "pm25_median": preds["p50"],
                "p10_lower": preds["p10"],
                "p90_upper": preds["p90"],
                "confidence_score": conf,
                "health_band": band["band"],
                "badge": band["badge"],
                "color": band["color"],
                "advisory": band["advisory"],
            }

            forecast_output["regions"][reg]["forecasts"].append(reg_data)
            hour_summary["regional_predictions"][reg] = reg_data

            sg_sum += preds["p50"]
            if preds["p50"] > h_max:
                h_max = preds["p50"]
            if preds["p50"] > max_forecast_across_sg:
                max_forecast_across_sg = preds["p50"]

        hour_summary["sg_average"] = round(sg_sum / len(SG_REGIONS), 1)
        hour_summary["max_pm25"] = round(h_max, 1)
        hour_summary["dominant_band"] = get_health_band(hour_summary["sg_average"])["band"]
        forecast_output["hourly_timeline"].append(hour_summary)

    # Derive overall athletic training advisory
    if max_forecast_across_sg <= 55.0:
        forecast_output["overall_advisory"] = (
            "✅ Safe to Train: All regions forecast below 55 µg/m³. Full outdoor training sessions can proceed normally."
        )
    elif max_forecast_across_sg <= 150.0:
        forecast_output["overall_advisory"] = (
            "⚠️ Caution: Some regions forecast elevated PM2.5 (56-150 µg/m³). Shorten high-intensity training or relocate to regions with lower readings."
        )
    else:
        forecast_output["overall_advisory"] = (
            "🛑 Unhealthy: Severe haze forecast (>150 µg/m³). Halt strenuous outdoor athletic sessions and move workouts indoors."
        )

    # Save to JSON
    LATEST_FORECAST_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LATEST_FORECAST_FILE, "w") as f:
        json.dump(forecast_output, f, indent=2)

    return forecast_output
