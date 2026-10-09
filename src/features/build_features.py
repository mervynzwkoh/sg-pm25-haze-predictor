"""Feature engineering pipeline for Singapore PM2.5 forecasting."""
import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional

from src.config import SG_REGIONS
from src.features.wind_geometry import decompose_wind, compute_transboundary_haze_risk

logger = logging.getLogger(__name__)


def add_time_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Add cyclical diurnal (hour-of-day) and calendar features."""
    df = df.copy()
    ts = pd.to_datetime(df[timestamp_col])

    # Hour of day (0-23) cyclical encoding
    hours = ts.dt.hour
    df["sin_hour"] = np.sin(2 * np.pi * hours / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * hours / 24.0)

    # Day of week (0-6) cyclical encoding
    dow = ts.dt.dayofweek
    df["sin_dow"] = np.sin(2 * np.pi * dow / 7.0)
    df["cos_dow"] = np.cos(2 * np.pi * dow / 7.0)

    # Morning and evening training flags
    df["is_morning_training_window"] = ((hours >= 6) & (hours <= 9)).astype(int)
    df["is_evening_training_window"] = ((hours >= 17) & (hours <= 20)).astype(int)

    return df


def add_wind_features(df: pd.DataFrame, speed_col: str = "wind_speed", dir_col: str = "wind_direction") -> pd.DataFrame:
    """Add decomposed wind vector components and meteorological wind characteristics."""
    df = df.copy()
    if speed_col in df.columns and dir_col in df.columns:
        u_vals = []
        v_vals = []
        for s, d in zip(df[speed_col], df[dir_col]):
            if pd.isna(s) or pd.isna(d):
                u_vals.append(0.0)
                v_vals.append(0.0)
            else:
                u, v = decompose_wind(float(s), float(d))
                u_vals.append(u)
                v_vals.append(v)
        df["wind_u"] = u_vals
        df["wind_v"] = v_vals
    return df


def add_autoregressive_features(
    df: pd.DataFrame,
    regions: List[str] = list(SG_REGIONS.keys()),
    lags: List[int] = [1, 2, 3, 6, 12, 24],
    rolling_windows: List[int] = [3, 6, 12, 24]
) -> pd.DataFrame:
    """Add autoregressive lags and rolling statistics for all regional PM2.5 series."""
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    new_cols = {}

    for reg in regions:
        col = f"pm25_{reg}"
        if col not in df.columns:
            continue

        # Lags
        for lag in lags:
            new_cols[f"{col}_lag_{lag}"] = df[col].shift(lag)

        # Rolling statistics (using closed='left' equivalent via shift(1) to avoid lookahead leak)
        shifted = df[col].shift(1)
        for w in rolling_windows:
            new_cols[f"{col}_roll_mean_{w}"] = shifted.rolling(w, min_periods=1).mean()
            new_cols[f"{col}_roll_std_{w}"] = shifted.rolling(w, min_periods=1).std().fillna(0.0)
            new_cols[f"{col}_roll_min_{w}"] = shifted.rolling(w, min_periods=1).min()
            new_cols[f"{col}_roll_max_{w}"] = shifted.rolling(w, min_periods=1).max()

        # Rates of change (momentum)
        new_cols[f"{col}_diff_1h"] = new_cols[f"{col}_lag_1"] - new_cols[f"{col}_lag_2"]
        new_cols[f"{col}_diff_3h"] = new_cols[f"{col}_lag_1"] - new_cols[f"{col}_lag_3"]

    # Spatial regional gradients
    if "pm25_west" in df.columns and "pm25_east" in df.columns:
        new_cols["grad_west_east"] = df["pm25_west"].shift(1) - df["pm25_east"].shift(1)
    if "pm25_south" in df.columns and "pm25_north" in df.columns:
        new_cols["grad_south_north"] = df["pm25_south"].shift(1) - df["pm25_north"].shift(1)

    new_df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return new_df


def add_transboundary_advection_features(
    df: pd.DataFrame,
    indonesia_readings: Dict[str, float]
) -> pd.DataFrame:
    """Add physical transboundary haze inflow potential based on Indonesian upstream monitors and wind vectors."""
    df = df.copy()

    adv_indices = []
    batam_risks = []
    palembang_risks = []
    jambi_risks = []

    for _, row in df.iterrows():
        w_speed = row.get("wind_speed", 3.0)
        w_dir = row.get("wind_direction", 180.0)
        if pd.isna(w_speed):
            w_speed = 3.0
        if pd.isna(w_dir):
            w_dir = 180.0

        risk_dict = compute_transboundary_haze_risk(indonesia_readings, float(w_speed), float(w_dir))
        adv_indices.append(risk_dict.get("transboundary_haze_index", 0.0))
        batam_risks.append(risk_dict.get("adv_risk_batam", 0.0))
        palembang_risks.append(risk_dict.get("adv_risk_talang_betutu_palembang", 0.0))
        jambi_risks.append(risk_dict.get("adv_risk_kota_jambi", 0.0))

    df["transboundary_haze_index"] = adv_indices
    df["adv_risk_batam"] = batam_risks
    df["adv_risk_palembang"] = palembang_risks
    df["adv_risk_jambi"] = jambi_risks

    # Dynamic hourly upstream corridor & atmospheric transport physics
    if "palembang_pm25" in df.columns:
        w_dir = df.get("wind_direction_10m", df.get("wind_direction", 180.0)).fillna(180.0)
        w_spd = df.get("wind_speed_10m", df.get("wind_speed", 3.5)).fillna(3.5)
        blow_to = (w_dir + 180.0) % 360.0
        angle_diff = (blow_to - 345.0).abs()
        angle_diff = np.where(angle_diff > 180.0, 360.0 - angle_diff, angle_diff)
        alignment = np.maximum(0.0, np.cos(np.radians(angle_diff)))

        df["dynamic_sumatra_inflow"] = df["palembang_pm25"] * alignment * (w_spd / 5.0)

        # Transport transit time lags (smoke travel time across Malacca Straits: 6h, 12h, 18h)
        for l in [6, 12, 18]:
            df[f"palembang_pm25_lag_{l}"] = df["palembang_pm25"].shift(l)
            df[f"dynamic_sumatra_inflow_lag_{l}"] = df["dynamic_sumatra_inflow"].shift(l)

    if "precipitation" in df.columns:
        df["rain_scavenging_3h"] = df["precipitation"].rolling(3, min_periods=1).sum().fillna(0.0)

    return df


def prepare_training_dataset(
    df: pd.DataFrame,
    horizons: List[int] = list(range(1, 13))
) -> Tuple[pd.DataFrame, Dict[str, Dict[int, pd.Series]], List[str]]:
    """Build feature matrix X and target delta dictionaries {region: {horizon: delta_series}}

    without future lookahead data leakage.
    Target represents physical change: Delta_h = y_{t+h} - y_t.
    """
    targets = {}
    feature_df = df.copy().sort_values("timestamp").reset_index(drop=True)

    for reg in SG_REGIONS.keys():
        targets[reg] = {}
        cur_series = feature_df[f"pm25_{reg}"]
        for h in horizons:
            # Target for horizon h: delta change from current observation y_t
            delta_col = f"target_delta_{reg}_h{h}"
            feature_df[delta_col] = cur_series.shift(-h) - cur_series
            targets[reg][h] = feature_df[delta_col]

    # Drop target columns and metadata to define feature column set
    exclude_cols = [
        "timestamp", "date", "updatedTimestamp", "ts_hour"
    ] + [
        f"target_delta_{r}_h{h}" for r in SG_REGIONS.keys() for h in horizons
    ]

    numeric_cols = feature_df.select_dtypes(include=[np.number, bool]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    return feature_df, targets, feature_cols
