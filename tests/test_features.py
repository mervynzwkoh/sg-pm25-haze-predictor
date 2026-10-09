"""Unit tests for wind geometry and feature engineering."""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from src.features.wind_geometry import (
    calculate_bearing,
    decompose_wind,
    calculate_advection_alignment,
    compute_transboundary_haze_risk,
)
from src.features.build_features import (
    add_time_features,
    add_wind_features,
    add_autoregressive_features,
    prepare_training_dataset,
)


def test_calculate_bearing():
    # Due North: lat 0, lon 0 -> lat 10, lon 0 should be 0 deg
    bearing = calculate_bearing(0.0, 0.0, 10.0, 0.0)
    assert pytest.approx(bearing, 0.1) == 0.0

    # Due East: lat 0, lon 0 -> lat 0, lon 10 should be 90 deg
    bearing_e = calculate_bearing(0.0, 0.0, 0.0, 10.0)
    assert pytest.approx(bearing_e, 0.1) == 90.0


def test_decompose_wind():
    # Wind from North (0 deg) blows South -> u=0, v=-speed
    u, v = decompose_wind(10.0, 0.0)
    assert pytest.approx(u, 0.01) == 0.0
    assert pytest.approx(v, 0.01) == -10.0

    # Wind from South (180 deg) blows North -> u=0, v=+speed
    u, v = decompose_wind(10.0, 180.0)
    assert pytest.approx(u, 0.01) == 0.0
    assert pytest.approx(v, 0.01) == 10.0


def test_advection_alignment():
    # If bearing to SG is 0 (North) and wind blows from South (180 deg, blowing North), alignment should be 1.0
    align = calculate_advection_alignment(180.0, 0.0)
    assert pytest.approx(align, 0.01) == 1.0

    # If wind blows away from SG (from North, 0 deg), alignment should be 0.0
    align_away = calculate_advection_alignment(0.0, 0.0)
    assert pytest.approx(align_away, 0.01) == 0.0


def test_compute_transboundary_haze_risk():
    sample_readings = {
        "Batam": 150.0,
        "Talang Betutu Palembang": 300.0,
        "Kota Jambi": 250.0,
    }
    # Southerly wind blowing towards Singapore
    risk = compute_transboundary_haze_risk(sample_readings, wind_speed=5.0, wind_direction_deg=180.0)
    assert "transboundary_haze_index" in risk
    assert risk["transboundary_haze_index"] > 0.0


def test_feature_pipeline_no_lookahead():
    # Mock continuous 50 hours of data
    dates = pd.date_range("2026-10-01", periods=50, freq="h")
    df = pd.DataFrame({
        "timestamp": dates,
        "pm25_north": np.linspace(20, 80, 50),
        "pm25_south": np.linspace(25, 85, 50),
        "pm25_east": np.linspace(15, 60, 50),
        "pm25_west": np.linspace(30, 95, 50),
        "pm25_central": np.linspace(22, 75, 50),
        "wind_speed": 4.0,
        "wind_direction": 160.0,
    })

    df_time = add_time_features(df)
    assert "sin_hour" in df_time.columns
    assert "cos_hour" in df_time.columns

    df_wind = add_wind_features(df_time)
    assert "wind_u" in df_wind.columns
    assert "wind_v" in df_wind.columns

    df_lags = add_autoregressive_features(df_wind)
    assert "pm25_north_lag_1" in df_lags.columns

    feature_df, targets, feature_cols = prepare_training_dataset(df_lags, horizons=[1, 2, 3])
    assert len(feature_cols) > 10
    # Ensure current timestamp is not in feature columns
    assert "timestamp" not in feature_cols
    assert "target_north_h1" not in feature_cols


def test_strict_causality_and_no_future_leakage():
    """Audit causality: mutating future data at t >= 26 must have zero impact on features at t = 25."""
    np.random.seed(42)
    n = 60
    dates = pd.date_range("2026-10-01", periods=n, freq="h")
    
    base_df = pd.DataFrame({
        "timestamp": dates,
        "pm25_north": np.random.uniform(20, 50, n),
        "pm25_south": np.random.uniform(20, 50, n),
        "pm25_east": np.random.uniform(20, 50, n),
        "pm25_west": np.random.uniform(20, 50, n),
        "pm25_central": np.random.uniform(20, 50, n),
        "wind_speed": np.random.uniform(1, 6, n),
        "wind_direction": np.random.uniform(0, 360, n),
        "palembang_pm25": np.random.uniform(50, 150, n),
        "precipitation": np.random.uniform(0, 5, n),
    })

    from src.features.build_features import add_transboundary_advection_features

    def build_full(df_in):
        t_df = add_time_features(df_in)
        w_df = add_wind_features(t_df)
        tr_df = add_transboundary_advection_features(w_df, {"Batam": 50.0, "Talang Betutu Palembang": 120.0})
        ar_df = add_autoregressive_features(tr_df)
        f_df, _, f_cols = prepare_training_dataset(ar_df, horizons=[1, 2])
        return f_df, f_cols

    feat_orig, f_cols = build_full(base_df)
    
    # Mutate future values at t >= 26
    corrupted_df = base_df.copy()
    corrupted_df.loc[26:, "pm25_north"] += 999.0
    corrupted_df.loc[26:, "pm25_central"] += 888.0
    corrupted_df.loc[26:, "precipitation"] += 50.0
    corrupted_df.loc[26:, "palembang_pm25"] += 777.0

    feat_corrupt, _ = build_full(corrupted_df)

    # Check row 25 (index 25)
    row_orig = feat_orig.loc[25, f_cols].astype(float)
    row_corrupt = feat_corrupt.loc[25, f_cols].astype(float)

    diff = np.abs(row_orig - row_corrupt)
    max_leakage = diff.max()
    assert max_leakage == 0.0, f"Leakage detected! Max diff at t=25 is {max_leakage}"

