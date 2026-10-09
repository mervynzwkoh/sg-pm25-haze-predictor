"""Comprehensive 4-tier ablation study and baseline comparison suite.

Evaluates:
- Baseline 1: Naive Persistence
- Baseline 2: Diurnal Seasonal Climatology
- Baseline 3: Regularized Autoregressive (Ridge AR)
- Baseline 4: Standard LightGBM (No Physics)
- Model M0: Local PM2.5 Autoregressive Lags Only
- Model M1: Local Lags + Local Meteorology (Temp, RH, Wind Speed, Pressure)
- Model M2: Local + Meteo + Upstream Telemetry (Palembang, Batam, Jambi)
- Model M3: Full Proposed Architecture (M2 + Advection Physics + Wind Alignment + Rain Scavenging)

Key Decision Criterion Evaluation:
Directly tests whether transboundary physics significantly improves 6-10h hazardous event detection (>55 µg/m³).
"""
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "2"
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import SG_REGIONS, FORECAST_HORIZONS, DATA_DIR, RAW_DATA_DIR
from src.features.build_features import (
    add_time_features,
    add_wind_features,
    add_autoregressive_features,
    add_transboundary_advection_features,
    prepare_training_dataset,
)
from src.data.indonesia_client import IndonesiaHazeData
from src.models.baselines import (
    PersistenceBaseline,
    DiurnalSeasonalBaseline,
    RidgeARBaseline,
    StandardLGBMBaseline,
)

logger = logging.getLogger(__name__)
REPORTS_DIR = DATA_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def build_ablation_features(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Engineer features and partition into distinct ablation feature groups."""
    df = df_raw.copy().sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("Asia/Singapore")

    if "wind_speed" not in df.columns:
        df["wind_speed"] = df.get("wind_speed_10m", 3.5)
    if "wind_direction" not in df.columns:
        df["wind_direction"] = df.get("wind_direction_10m", 180.0)

    indo_data = IndonesiaHazeData()
    upstream_readings = indo_data.get_station_readings()

    df_time = add_time_features(df)
    df_wind = add_wind_features(df_time)
    df_trans = add_transboundary_advection_features(df_wind, upstream_readings)
    df_feat = add_autoregressive_features(df_trans)

    # Define feature groups:
    # 1. Local lags only (M0)
    local_lag_cols = [c for c in df_feat.columns if any(c.startswith(f"pm25_{r}_lag_") or c.startswith(f"pm25_{r}_roll_") for r in SG_REGIONS.keys()) or c in ["grad_west_east", "grad_south_north"]]

    # 2. Local lags + local meteorology + diurnal time (M1)
    meteo_cols = ["temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed", "wind_direction", "sin_hour", "cos_hour", "sin_dow", "cos_dow"]
    meteo_cols = [c for c in meteo_cols if c in df_feat.columns]
    m1_cols = local_lag_cols + meteo_cols

    # 3. M1 + Upstream Telemetry (M2)
    upstream_raw_cols = ["palembang_pm25", "palembang_co", "palembang_aod"]
    upstream_raw_cols = [c for c in upstream_raw_cols if c in df_feat.columns]
    m2_cols = m1_cols + upstream_raw_cols

    # 4. Full Physics-Informed (M3)
    physics_cols = [
        "wind_u", "wind_v", "transboundary_haze_index", "adv_risk_batam", "adv_risk_palembang", "adv_risk_jambi",
        "dynamic_sumatra_inflow", "palembang_pm25_lag_6", "dynamic_sumatra_inflow_lag_6",
        "palembang_pm25_lag_12", "dynamic_sumatra_inflow_lag_12", "palembang_pm25_lag_18", "dynamic_sumatra_inflow_lag_18",
        "rain_scavenging_3h"
    ]
    physics_cols = [c for c in physics_cols if c in df_feat.columns]
    m3_cols = m2_cols + physics_cols

    feature_groups = {
        "M0_local_only": local_lag_cols,
        "M1_local_meteo": m1_cols,
        "M2_with_upstream": m2_cols,
        "M3_full_physics": m3_cols,
    }

    return df_feat, feature_groups


def run_ablation_benchmark(dataset_csv: Path = RAW_DATA_DIR / "multi_year_haze_seasons_2022_2026.csv") -> Dict[str, Any]:
    """Execute complete ablation study and baseline comparison."""
    logger.info(f"Loading dataset from {dataset_csv}...")
    df_raw = pd.read_csv(dataset_csv)
    df_feat, feature_groups = build_ablation_features(df_raw)

    # Temporal split: Train on 2022 to Sep 30, 2026; Holdout test on the October 2026 severe haze surge
    split_date = "2026-10-01"
    train_mask = (df_feat["timestamp"] < split_date) & (df_feat["pm25_central_lag_24"].notna())
    test_mask = (df_feat["timestamp"] >= split_date)

    df_train = df_feat[train_mask].reset_index(drop=True)
    df_test = df_feat[test_mask].reset_index(drop=True)

    logger.info(f"Training samples: {len(df_train)} | Test holdout samples: {len(df_test)}")

    # Fit Diurnal Seasonal Baseline on training data
    diurnal_model = DiurnalSeasonalBaseline()
    diurnal_model.fit(df_train)

    horizons = list(range(1, 13))
    results = []

    # Hazard detection contingency accumulators for key horizons h in [6..10]
    hazard_threshold = 55.0  # Elevated haze threshold in Singapore
    hazard_events = {
        "persistence": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        "diurnal": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        "ridge_ar": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        "M0_local_only": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        "M1_local_meteo": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        "M2_with_upstream": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        "M3_full_physics": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
    }

    test_hours = pd.to_datetime(df_test["timestamp"]).dt.hour.values

    for h in horizons:
        for reg in SG_REGIONS.keys():
            col = f"pm25_{reg}"
            y_base_train = df_train[col].values
            y_true_train_future = df_train[col].shift(-h).values
            valid_train = ~np.isnan(y_true_train_future)
            delta_train = y_true_train_future[valid_train] - y_base_train[valid_train]

            y_base_test = df_test[col].values
            y_true_test = df_test[col].shift(-h).values
            valid_test = ~np.isnan(y_true_test)

            if np.sum(valid_test) == 0:
                continue

            y_test_eval = y_true_test[valid_test]
            base_eval = y_base_test[valid_test]
            test_hrs_eval = test_hours[valid_test]

            # 1. Baseline: Naive Persistence
            pred_persistence = base_eval
            mae_pers = float(mean_absolute_error(y_test_eval, pred_persistence))
            rmse_pers = float(np.sqrt(mean_squared_error(y_test_eval, pred_persistence)))

            # 2. Baseline: Diurnal Climatology
            pred_diurnal = np.array([diurnal_model.predict(b, reg, h, hr) for b, hr in zip(base_eval, test_hrs_eval)])
            mae_diurnal = float(mean_absolute_error(y_test_eval, pred_diurnal))
            rmse_diurnal = float(np.sqrt(mean_squared_error(y_test_eval, pred_diurnal)))

            # 3. Baseline: Ridge AR
            ridge = Ridge(alpha=1.0)
            X_tr_lags = df_train.loc[valid_train, feature_groups["M0_local_only"]].fillna(0.0)
            X_te_lags = df_test.loc[valid_test, feature_groups["M0_local_only"]].fillna(0.0)
            ridge.fit(X_tr_lags, delta_train)
            pred_ridge = np.maximum(5.0, base_eval + ridge.predict(X_te_lags))
            mae_ridge = float(mean_absolute_error(y_test_eval, pred_ridge))
            rmse_ridge = float(np.sqrt(mean_squared_error(y_test_eval, pred_ridge)))

            # 4. Ablations M0, M1, M2, M3 using LightGBM
            ablation_preds = {}
            for m_name, f_cols in feature_groups.items():
                X_tr = df_train.loc[valid_train, f_cols].fillna(0.0)
                X_te = df_test.loc[valid_test, f_cols].fillna(0.0)

                lgbm = LGBMRegressor(
                    objective="quantile",
                    alpha=0.50,
                    n_estimators=50,
                    learning_rate=0.04,
                    num_leaves=18,
                    min_child_samples=8,
                    n_jobs=4,
                    verbosity=-1,
                    random_state=42
                )
                lgbm.fit(X_tr, delta_train)
                delta_pred = lgbm.predict(X_te)
                pred_abs = np.maximum(5.0, base_eval + delta_pred)
                ablation_preds[m_name] = pred_abs

            mae_m0 = float(mean_absolute_error(y_test_eval, ablation_preds["M0_local_only"]))
            mae_m1 = float(mean_absolute_error(y_test_eval, ablation_preds["M1_local_meteo"]))
            mae_m2 = float(mean_absolute_error(y_test_eval, ablation_preds["M2_with_upstream"]))
            mae_m3 = float(mean_absolute_error(y_test_eval, ablation_preds["M3_full_physics"]))

            rmse_m0 = float(np.sqrt(mean_squared_error(y_test_eval, ablation_preds["M0_local_only"])))
            rmse_m1 = float(np.sqrt(mean_squared_error(y_test_eval, ablation_preds["M1_local_meteo"])))
            rmse_m2 = float(np.sqrt(mean_squared_error(y_test_eval, ablation_preds["M2_with_upstream"])))
            rmse_m3 = float(np.sqrt(mean_squared_error(y_test_eval, ablation_preds["M3_full_physics"])))

            results.append({
                "horizon_h": h,
                "region": reg,
                "persistence_mae": round(mae_pers, 2),
                "diurnal_mae": round(mae_diurnal, 2),
                "ridge_ar_mae": round(mae_ridge, 2),
                "m0_local_mae": round(mae_m0, 2),
                "m1_meteo_mae": round(mae_m1, 2),
                "m2_upstream_mae": round(mae_m2, 2),
                "m3_full_physics_mae": round(mae_m3, 2),
                "persistence_rmse": round(rmse_pers, 2),
                "m0_local_rmse": round(rmse_m0, 2),
                "m3_full_physics_rmse": round(rmse_m3, 2),
                "skill_gain_m3_vs_persistence": round((1.0 - mae_m3 / max(0.01, mae_pers)) * 100.0, 2),
                "skill_gain_m3_vs_m0": round((1.0 - mae_m3 / max(0.01, mae_m0)) * 100.0, 2),
            })

            # Accumulate contingency stats for 6-10h horizons
            if 6 <= h <= 10:
                y_act_haz = (y_test_eval > hazard_threshold)
                models_to_score = {
                    "persistence": pred_persistence,
                    "diurnal": pred_diurnal,
                    "ridge_ar": pred_ridge,
                    "M0_local_only": ablation_preds["M0_local_only"],
                    "M1_local_meteo": ablation_preds["M1_local_meteo"],
                    "M2_with_upstream": ablation_preds["M2_with_upstream"],
                    "M3_full_physics": ablation_preds["M3_full_physics"],
                }
                for m_k, p_vals in models_to_score.items():
                    p_haz = (p_vals > hazard_threshold)
                    tp = int(np.sum(p_haz & y_act_haz))
                    fp = int(np.sum(p_haz & ~y_act_haz))
                    fn = int(np.sum(~p_haz & y_act_haz))
                    tn = int(np.sum(~p_haz & ~y_act_haz))
                    hazard_events[m_k]["tp"] += tp
                    hazard_events[m_k]["fp"] += fp
                    hazard_events[m_k]["fn"] += fn
                    hazard_events[m_k]["tn"] += tn

        logger.info(f"--- Completed evaluating all regions for Horizon h={h}/12 ---")

    res_df = pd.DataFrame(results)
    res_df.to_csv(REPORTS_DIR / "ablation_benchmark.csv", index=False)

    # Compute Contingency Metrics for Key Decision Criterion
    hazard_metrics = []
    for m_k, c in hazard_events.items():
        tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
        recall = tp / max(1, (tp + fn)) * 100.0
        precision = tp / max(1, (tp + fp)) * 100.0
        far = fp / max(1, (tp + fp)) * 100.0
        csi = tp / max(1, (tp + fp + fn)) * 100.0  # Critical Success Index
        hazard_metrics.append({
            "model": m_k,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "recall_pct": round(recall, 2),
            "precision_pct": round(precision, 2),
            "false_alarm_ratio_pct": round(far, 2),
            "critical_success_index_csi_pct": round(csi, 2),
        })

    haz_df = pd.DataFrame(hazard_metrics)
    haz_df.to_csv(REPORTS_DIR / "hazard_event_detection_6_10h.csv", index=False)

    logger.info("=== ABLATION & KEY DECISION CRITERION COMPLETE ===")
    print("\n--- Key Decision Criterion (6-10h Hazard Detection >55 µg/m³) ---")
    print(haz_df.to_string(index=False))

    return {"ablation_df": res_df, "hazard_df": haz_df}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_ablation_benchmark()
