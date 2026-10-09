"""Comprehensive validation benchmark comparing model architectures on historical haze data.

Evaluates on the October 2026 severe haze surge holdout set, computes MAE/RMSE/PICP,
and generates publication-quality validation plots.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import SG_REGIONS, DATA_DIR, RAW_DATA_DIR
from src.features.build_features import (
    add_time_features,
    add_wind_features,
    add_autoregressive_features,
    add_transboundary_advection_features,
)
from src.data.indonesia_client import IndonesiaHazeData

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPORTS_DIR = DATA_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def build_haze_season_features(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Build full feature set on haze season dataset."""
    df = df_raw.copy().sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("Asia/Singapore")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Singapore")

    # Fetch real 50-day weather (rain, wind speed, wind dir) and upstream smoke corridor from Open-Meteo
    from src.data.openmeteo_client import OpenMeteoClient
    meteo_client = OpenMeteoClient()
    df_w = meteo_client.fetch_singapore_meteo(past_days=52, forecast_days=2)
    df_smoke = meteo_client.fetch_upstream_smoke_corridor(past_days=52, forecast_days=2)

    if not df_w.empty:
        df["ts_h"] = df["timestamp"].dt.floor("h")
        df_w["ts_h"] = df_w["timestamp"].dt.floor("h")
        df = pd.merge(df, df_w.drop(columns=["timestamp"]), on="ts_h", how="left")
    if not df_smoke.empty:
        df_smoke["ts_h"] = df_smoke["timestamp"].dt.floor("h")
        df = pd.merge(df, df_smoke.drop(columns=["timestamp"]), on="ts_h", how="left")

    if "wind_speed" not in df.columns:
        df["wind_speed"] = df.get("wind_speed_10m", 3.5)
    if "wind_direction" not in df.columns:
        df["wind_direction"] = df.get("wind_direction_10m", 180.0)

    # Indonesian upstream readings proxy
    indo_data = IndonesiaHazeData()
    upstream_readings = indo_data.get_station_readings()

    df_time = add_time_features(df)
    df_wind = add_wind_features(df_time)
    df_trans = add_transboundary_advection_features(df_wind, upstream_readings)
    df_feat = add_autoregressive_features(df_trans)

    exclude_cols = [
        "timestamp", "date", "updatedTimestamp", "ts_hour", "ts_h"
    ] + [f"pm25_{r}" for r in SG_REGIONS.keys()]

    feature_cols = [c for c in df_feat.select_dtypes(include=[np.number, bool]).columns if c not in exclude_cols]
    return df_feat, feature_cols


def run_validation_benchmark(
    csv_path: Path = RAW_DATA_DIR / "sg_pm25_haze_season_2026.csv",
    split_date: str = "2026-10-01"
) -> Dict[str, Any]:
    """Execute rigorous out-of-time validation on held-out October surge."""
    logger.info(f"Loading haze season dataset from {csv_path}...")
    df_raw = pd.read_csv(csv_path)
    df_feat, feature_cols = build_haze_season_features(df_raw)

    train_mask = (df_feat["timestamp"] < split_date) & (df_feat["pm25_central_lag_24"].notna())
    val_mask = (df_feat["timestamp"] >= split_date)

    logger.info(f"Train period: {df_feat.loc[train_mask, 'timestamp'].iloc[0]} to {df_feat.loc[train_mask, 'timestamp'].iloc[-1]} ({train_mask.sum()} hours)")
    logger.info(f"Validation period: {df_feat.loc[val_mask, 'timestamp'].iloc[0]} to {df_feat.loc[val_mask, 'timestamp'].iloc[-1]} ({val_mask.sum()} hours)")

    horizons = list(range(1, 13))
    results = []
    trajectories = {}

    for reg in SG_REGIONS.keys():
        trajectories[reg] = {
            "timestamps": [ts.isoformat() for ts in df_feat.loc[val_mask, "timestamp"]],
            "actual": df_feat.loc[val_mask, f"pm25_{reg}"].tolist(),
            "pred_adaptive": {},
            "pred_direct": {},
            "p10": {},
            "p90": {},
        }

    for h in horizons:
        for reg in SG_REGIONS.keys():
            col = f"pm25_{reg}"
            cur_series = df_feat[col]
            target_series = cur_series.shift(-h)
            delta_series = target_series - cur_series

            tr_idx = train_mask & target_series.notna()
            val_idx = val_mask & target_series.notna()

            X_tr = df_feat.loc[tr_idx, feature_cols].fillna(0.0)
            y_tr_abs = target_series.loc[tr_idx]
            y_tr_delta = delta_series.loc[tr_idx]

            X_v = df_feat.loc[val_idx, feature_cols].fillna(0.0)
            y_v_true = target_series.loc[val_idx].values
            y_v_base = cur_series.loc[val_idx].values

            # Baseline Naive Model (Persistence: y_{t+h} = y_t)
            pred_naive = y_v_base
            mae_naive = float(mean_absolute_error(y_v_true, pred_naive))
            rmse_naive = float(np.sqrt(mean_squared_error(y_v_true, pred_naive)))
            mape_naive = float(np.mean(np.abs((y_v_true - pred_naive) / np.maximum(y_v_true, 1.0))) * 100.0)

            # Adaptive Model (Quantiles p10, p50, p90)
            m_delta_50 = LGBMRegressor(objective="quantile", alpha=0.50, n_estimators=85, learning_rate=0.04, num_leaves=18, min_child_samples=8, verbosity=-1, random_state=42)
            m_delta_50.fit(X_tr, y_tr_delta)
            delta_pred_50 = m_delta_50.predict(X_v)
            pred_model = np.maximum(5.0, y_v_base + delta_pred_50)

            # Quantiles p10 and p90
            m_delta_10 = LGBMRegressor(objective="quantile", alpha=0.10, n_estimators=85, learning_rate=0.04, num_leaves=18, min_child_samples=8, verbosity=-1, random_state=42)
            m_delta_10.fit(X_tr, y_tr_delta)
            p10 = np.maximum(5.0, y_v_base + m_delta_10.predict(X_v))

            m_delta_90 = LGBMRegressor(objective="quantile", alpha=0.90, n_estimators=85, learning_rate=0.04, num_leaves=18, min_child_samples=8, verbosity=-1, random_state=42)
            m_delta_90.fit(X_tr, y_tr_delta)
            p90 = np.maximum(pred_model, y_v_base + m_delta_90.predict(X_v))

            mae_model = float(mean_absolute_error(y_v_true, pred_model))
            rmse_model = float(np.sqrt(mean_squared_error(y_v_true, pred_model)))
            mape_model = float(np.mean(np.abs((y_v_true - pred_model) / np.maximum(y_v_true, 1.0))) * 100.0)
            coverage = float(np.mean((y_v_true >= p10) & (y_v_true <= p90)) * 100.0)
            interval_width = float(np.mean(p90 - p10))
            skill_gain = float((1.0 - mae_model / max(0.01, mae_naive)) * 100.0)

            results.append({
                "horizon_h": h,
                "region": reg,
                "model_mae": round(mae_model, 2),
                "model_rmse": round(rmse_model, 2),
                "model_mape": round(mape_model, 2),
                "naive_mae": round(mae_naive, 2),
                "naive_rmse": round(rmse_naive, 2),
                "naive_mape": round(mape_naive, 2),
                "skill_gain_pct": round(skill_gain, 2),
                "coverage_80pct": round(coverage, 1),
                "interval_width_ugm3": round(interval_width, 2),
            })

            # Record trajectory for horizon h=3 and h=6
            if h in [1, 3, 6]:
                trajectories[reg]["pred_adaptive"][f"h{h}"] = [round(float(x), 1) for x in pred_model]
                trajectories[reg]["p10"][f"h{h}"] = [round(float(x), 1) for x in p10]
                trajectories[reg]["p90"][f"h{h}"] = [round(float(x), 1) for x in p90]

    res_df = pd.DataFrame(results)
    res_df.to_csv(REPORTS_DIR / "validation_summary.csv", index=False)

    # Generate Publication Plots
    plot_validation_results(res_df, trajectories, df_feat.loc[val_mask, "timestamp"].values)

    return {"summary_df": res_df, "trajectories": trajectories}


def plot_validation_results(res_df: pd.DataFrame, trajectories: Dict[str, Any], val_timestamps: np.ndarray):
    """Render publication-quality validation plots focused on forecast accuracy across horizons."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    h_summary = res_df.groupby("horizon_h")[["model_mae", "naive_mae", "model_rmse", "naive_rmse", "model_mape", "naive_mape", "coverage_80pct", "interval_width_ugm3"]].mean()

    # 1. Forecast Error (MAE & RMSE) vs Time Horizon
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    ax1.plot(h_summary.index, h_summary["model_mae"], "o-", color="#1f77b4", linewidth=2.5, label="PM2.5 Forecasting Model")
    ax1.plot(h_summary.index, h_summary["naive_mae"], "s--", color="#7f7f7f", linewidth=2.0, label="Baseline Naive (Persistence)")
    ax1.set_title("Mean Absolute Error (MAE) vs Forecast Horizon", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Forecast Horizon (Hours Ahead)", fontsize=11)
    ax1.set_ylabel("MAE (µg/m³)", fontsize=11)
    ax1.set_xticks(range(1, 13))
    ax1.legend(fontsize=10, loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(h_summary.index, h_summary["model_rmse"], "o-", color="#2ca02c", linewidth=2.5, label="PM2.5 Forecasting Model")
    ax2.plot(h_summary.index, h_summary["naive_rmse"], "s--", color="#7f7f7f", linewidth=2.0, label="Baseline Naive (Persistence)")
    ax2.set_title("Root Mean Squared Error (RMSE) vs Forecast Horizon", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Forecast Horizon (Hours Ahead)", fontsize=11)
    ax2.set_ylabel("RMSE (µg/m³)", fontsize=11)
    ax2.set_xticks(range(1, 13))
    ax2.legend(fontsize=10, loc="upper left")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "forecast_accuracy_by_horizon.png", dpi=200)
    plt.close(fig)

    # 2. MAPE (%) and 80% Coverage Calibration
    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(15, 5.5))

    ax3.plot(h_summary.index, h_summary["model_mape"], "d-", color="#d62728", linewidth=2.5, label="Model MAPE (%)")
    ax3.plot(h_summary.index, h_summary["naive_mape"], "s--", color="#7f7f7f", linewidth=2.0, label="Naive Baseline MAPE (%)")
    ax3.set_title("Percentage Error (MAPE %) vs Forecast Horizon", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Forecast Horizon (Hours Ahead)", fontsize=11)
    ax3.set_ylabel("Mean Absolute Percentage Error (%)", fontsize=11)
    ax3.set_xticks(range(1, 13))
    ax3.legend(fontsize=10, loc="upper left")
    ax3.grid(True, linestyle="--", alpha=0.6)

    ax4.plot(h_summary.index, h_summary["coverage_80pct"], "o-", color="#9467bd", linewidth=2.5, label="Empirical Coverage")
    ax4.axhline(80.0, color="black", linestyle="--", linewidth=1.5, label="Nominal 80% Confidence Target")
    ax4.set_title("80% Prediction Interval Calibration vs Horizon", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Forecast Horizon (Hours Ahead)", fontsize=11)
    ax4.set_ylabel("Observations Covered (%)", fontsize=11)
    ax4.set_ylim([40, 100])
    ax4.set_xticks(range(1, 13))
    ax4.legend(fontsize=10, loc="lower left")
    ax4.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "forecast_uncertainty_and_coverage.png", dpi=200)
    plt.close(fig)

    # 3. Regional Accuracy Breakdown Plot
    fig, ax5 = plt.subplots(figsize=(11, 5.5))
    reg_colors = {"central": "#1f77b4", "west": "#d62728", "south": "#ff7f0e", "east": "#9467bd", "north": "#2ca02c"}

    for reg_key in SG_REGIONS.keys():
        reg_df = res_df[res_df["region"] == reg_key].set_index("horizon_h")
        ax5.plot(reg_df.index, reg_df["model_mae"], "o-", color=reg_colors[reg_key], linewidth=2.0, label=f"{SG_REGIONS[reg_key]['name']} Region")

    ax5.set_title("Forecast Accuracy (MAE) Across Singapore Planning Regions", fontsize=12, fontweight="bold")
    ax5.set_xlabel("Forecast Horizon (Hours Ahead)", fontsize=11)
    ax5.set_ylabel("MAE (µg/m³)", fontsize=11)
    ax5.set_xticks(range(1, 13))
    ax5.legend(fontsize=10, loc="upper left")
    ax5.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "regional_accuracy_breakdown.png", dpi=200)
    plt.close(fig)

    # 4. Actual vs 3-Hour Ahead Forecast Trajectory on October Surge
    fig, ax6 = plt.subplots(figsize=(14, 5.5))
    central_data = trajectories["central"]
    ts = pd.to_datetime(central_data["timestamps"])
    actual = central_data["actual"]
    pred_adapt = central_data["pred_adaptive"]["h3"]
    p10 = central_data["p10"]["h3"]
    p90 = central_data["p90"]["h3"]

    n_pts = len(pred_adapt)
    ts_pts = ts[:n_pts]
    act_pts = actual[:n_pts]

    ax6.plot(ts_pts, act_pts, color="black", linewidth=2.2, label="Actual Ground Truth (Central SG)")
    ax6.plot(ts_pts, pred_adapt, color="#1f77b4", linewidth=2.0, label="Model Forecast (+3 Hours Ahead)")
    ax6.fill_between(ts_pts, p10, p90, color="#1f77b4", alpha=0.22, label="80% Prediction Interval [p10, p90]")

    ax6.axhline(55, color="orange", linestyle=":", label="Elevated Threshold (55 µg/m³)")
    ax6.axhline(150, color="red", linestyle=":", label="Unhealthy Threshold (150 µg/m³)")

    ax6.set_title("Validation: 3-Hour Ahead Forecast Tracking the October 2026 Severe Haze Surge", fontsize=12, fontweight="bold")
    ax6.set_ylabel("PM2.5 Concentration (µg/m³)", fontsize=11)
    ax6.set_xlabel("Date (October 2026)", fontsize=11)
    ax6.legend(loc="upper left", fontsize=10)
    ax6.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "actual_vs_forecast_october_surge.png", dpi=200)
    plt.close(fig)

    logger.info(f"Accuracy validation plots successfully generated in {REPORTS_DIR}")

    logger.info(f"Validation plots successfully written to {REPORTS_DIR}")


if __name__ == "__main__":
    run_validation_benchmark()
