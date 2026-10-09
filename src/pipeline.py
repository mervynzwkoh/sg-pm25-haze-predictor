"""End-to-end automated pipeline for data ingestion, model training/inference,

and notification dispatch.
"""
import argparse
import logging
from datetime import datetime
import pandas as pd
from pytz import timezone

from src.config import SG_REGIONS, DATA_DIR, RAW_DATA_DIR
from src.data.datagov_client import DataGovClient
from src.data.openmeteo_client import OpenMeteoClient
from src.data.indonesia_client import IndonesiaHazeData
from src.data.google_aqi_client import GoogleAirQualityClient
from src.features.build_features import (
    add_time_features,
    add_wind_features,
    add_autoregressive_features,
    add_transboundary_advection_features,
    prepare_training_dataset,
)
from src.models.train import MultiHorizonForecaster, evaluate_forecast_horizons
from src.models.predict import generate_12h_forecast
from src.notifications.telegram_bot import TelegramNotifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
SGT = timezone("Asia/Singapore")


def run_pipeline(force_notify: bool = False, fetch_historical_days: int = 21):
    """Execute complete data pipeline, forecasting, and notification flow."""
    logger.info("=== Starting Singapore PM2.5 Forecasting Pipeline ===")

    # 1. Initialize Clients
    datagov = DataGovClient()
    openmeteo = OpenMeteoClient()
    indo_data = IndonesiaHazeData()
    google_aqi = GoogleAirQualityClient()
    notifier = TelegramNotifier()

    # 2. Ingest Latest Readings
    logger.info("Fetching real-time readings from data.gov.sg...")
    latest_pm25 = datagov.fetch_latest_pm25()
    latest_wind = datagov.fetch_latest_wind()
    current_readings = latest_pm25.get("readings", {})
    logger.info(f"Current regional PM2.5 readings: {current_readings}")

    # Fallback if live reading is empty
    if not current_readings:
        current_readings = {reg: 30.0 for reg in SG_REGIONS.keys()}

    # Check Google AQI (if available)
    if google_aqi.is_available:
        g_data = google_aqi.fetch_current_conditions()
        logger.info(f"Google AQI response: {g_data.get('google_pm25')}")

    # Upstream Indonesian station levels
    upstream_readings = indo_data.get_station_readings()
    regional_summary = indo_data.get_regional_summary()
    logger.info(f"Indonesian Upstream Haze Summary: {regional_summary}")

    # 3. Build / Load Training Historical Data (Haze season from late August onwards)
    haze_season_file = RAW_DATA_DIR / "sg_pm25_haze_season_2026.csv"
    df_history = pd.DataFrame()

    if haze_season_file.exists():
        logger.info(f"Loading full haze season dataset from {haze_season_file}")
        df_history = pd.read_csv(haze_season_file)
        df_history["timestamp"] = pd.to_datetime(df_history["timestamp"])
    else:
        logger.info(f"Fetching past {fetch_historical_days} days from data.gov.sg...")
        df_history = datagov.fetch_historical_pm25_range(days=fetch_historical_days)
        if not df_history.empty:
            df_history.to_csv(haze_season_file, index=False)

    # Augment with Open-Meteo weather and air quality features
    logger.info("Fetching Open-Meteo historical and forecast weather...")
    meteo_df = openmeteo.fetch_singapore_meteo(past_days=fetch_historical_days, forecast_days=2)
    aqi_df = openmeteo.fetch_singapore_air_quality(past_days=fetch_historical_days, forecast_days=2)
    smoke_df = openmeteo.fetch_upstream_smoke_corridor(past_days=fetch_historical_days, forecast_days=2)

    # Base grid from meteo_df (continuous hourly sequence)
    if not meteo_df.empty:
        meteo_df["ts_hour"] = meteo_df["timestamp"].dt.floor("h")
        base_df = meteo_df.copy()
        if not aqi_df.empty:
            aqi_df["ts_hour"] = aqi_df["timestamp"].dt.floor("h")
            base_df = pd.merge(base_df, aqi_df.drop(columns=["timestamp"]), on="ts_hour", how="left")
        if not smoke_df.empty:
            smoke_df["ts_hour"] = smoke_df["timestamp"].dt.floor("h")
            base_df = pd.merge(base_df, smoke_df.drop(columns=["timestamp"]), on="ts_hour", how="left")

        if not df_history.empty:
            df_history["ts_hour"] = df_history["timestamp"].dt.tz_convert("Asia/Singapore").dt.floor("h") if df_history["timestamp"].dt.tz is not None else df_history["timestamp"].dt.tz_localize("Asia/Singapore").dt.floor("h")
            df_merged = pd.merge(base_df, df_history.drop(columns=["timestamp"]), on="ts_hour", how="left")
        else:
            df_merged = base_df.copy()

        # If data.gov.sg regional PM2.5 has gaps or missing days, backfill with Open-Meteo AQI scaled to regions
        if "openmeteo_pm25" in df_merged.columns:
            regional_scalers = {"north": 0.95, "south": 1.05, "west": 1.08, "east": 0.92, "central": 1.00}
            for reg in SG_REGIONS.keys():
                col = f"pm25_{reg}"
                scaler = regional_scalers.get(reg, 1.0)
                if col not in df_merged.columns:
                    df_merged[col] = df_merged["openmeteo_pm25"] * scaler
                else:
                    df_merged[col] = df_merged[col].fillna(df_merged["openmeteo_pm25"] * scaler)
    else:
        df_merged = df_history.copy()

    # Align timestamp column
    if "ts_hour" in df_merged.columns:
        df_merged["timestamp"] = df_merged["ts_hour"]

    # Fill weather columns if missing
    if "wind_speed" not in df_merged.columns:
        df_merged["wind_speed"] = df_merged.get("wind_speed_10m", 3.0)
    if "wind_direction" not in df_merged.columns:
        df_merged["wind_direction"] = df_merged.get("wind_direction_10m", 180.0)

    # 4. Feature Engineering
    logger.info("Engineering features (wind geometry, lags, transboundary advection)...")
    df_feat = add_time_features(df_merged)
    df_feat = add_wind_features(df_feat)
    df_feat = add_transboundary_advection_features(df_feat, upstream_readings)
    df_feat = add_autoregressive_features(df_feat)

    # Prepare training targets and features
    feature_df, targets, feature_cols = prepare_training_dataset(df_feat)

    # 5. Temporal Train/Test Split (Walk-forward: 80% train, 20% test)
    n_samples = len(feature_df)
    split_idx = int(n_samples * 0.80)
    df_train = feature_df.iloc[:split_idx]
    df_test = feature_df.iloc[split_idx:]

    targets_train = {reg: {h: targets[reg][h].iloc[:split_idx] for h in targets[reg]} for reg in SG_REGIONS.keys()}
    targets_test = {reg: {h: targets[reg][h].iloc[split_idx:] for h in targets[reg]} for reg in SG_REGIONS.keys()}

    # 6. Train Models
    logger.info("Training LightGBM multi-horizon quantile models...")
    forecaster = MultiHorizonForecaster()
    forecaster.train(df_train, targets_train, feature_cols)

    # 7. Evaluate Performance Across Horizons
    eval_df = evaluate_forecast_horizons(forecaster, df_test, targets_test, df_test)
    if not eval_df.empty:
        summary_by_h = eval_df.groupby("horizon_h")[["adaptive_mae", "persistence_mae", "skill_gain_pct", "coverage_80pct"]].mean()
        logger.info("\n=== Multi-Horizon Model Evaluation Summary ===")
        print(summary_by_h.to_string())

    # 8. Real-time Inference for Next 12 Hours
    logger.info("Generating 12-hour real-time forecasts across Singapore...")
    latest_row = df_feat.iloc[[-1]]
    forecast_result = generate_12h_forecast(forecaster, latest_row, current_readings)

    # 9. Telegram Notifications
    logger.info("Checking notification criteria...")
    notifier.check_and_notify(forecast_result, force=force_notify)

    logger.info("Pipeline execution complete! Forecast written to data/latest_forecast.json")
    return forecast_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Singapore PM2.5 Forecast Pipeline")
    parser.add_argument("--force-notify", action="store_true", help="Force Telegram notification delivery")
    parser.add_argument("--days", type=int, default=21, help="Historical days of data to ingest")
    args = parser.parse_args()

    run_pipeline(force_notify=args.force_notify, fetch_historical_days=args.days)
