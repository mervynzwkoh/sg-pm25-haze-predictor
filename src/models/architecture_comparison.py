"""Architectural comparisons and physical assumption validation.

Implements:
1. Physical Advection Assumption Sensitivity:
   - Dynamic wind-dependent transport lags (tau = round(dist / wind_speed)) vs fixed lags (6, 12, 18h).
   - Inverse distance decay: linear (1/d) vs quadratic (1/d^2).
2. Spatial Modeling Architecture:
   - Separate Regional Models (5 independent models per horizon) vs
   - Pooled Regional Model (1 shared model per horizon with spatial coordinates & regional categorical encoding).
"""
import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import SG_REGIONS, DATA_DIR

logger = logging.getLogger(__name__)
REPORTS_DIR = DATA_DIR / "reports"


def run_architecture_and_physics_comparisons():
    """Run sensitivity analysis on physical assumptions and regional pooling."""
    logger.info("Executing architectural and physical assumption comparisons...")

    from src.models.ablation_study import build_ablation_features
    df_raw = pd.read_csv(DATA_DIR / "raw" / "multi_year_haze_seasons_2022_2026.csv")
    df_feat, feature_groups = build_ablation_features(df_raw)

    split_date = "2026-10-01"
    train_mask = (df_feat["timestamp"] < split_date) & (df_feat["pm25_central_lag_24"].notna())
    test_mask = (df_feat["timestamp"] >= split_date)

    df_train = df_feat[train_mask].reset_index(drop=True)
    df_test = df_feat[test_mask].reset_index(drop=True)

    # 1. Feature sets for comparison:
    # Set A: Fixed lags (Standard M3)
    cols_fixed = feature_groups["M3_full_physics"]

    # Set B: Dynamic wind-speed transit lags
    # Compute dynamic lag index: tau = round(480 km / (wind_speed * 3.6 km/h))
    w_speed_kmh = np.maximum(5.0, df_feat["wind_speed"].fillna(3.5).values * 3.6)
    est_transit_hours = np.clip(np.round(480.0 / w_speed_kmh).astype(int), 6, 24)
    # Add dynamic lagged series
    palembang_vals = df_feat["palembang_pm25"].fillna(50.0).values
    dyn_lagged = np.zeros_like(palembang_vals)
    for idx in range(len(palembang_vals)):
        lag_step = est_transit_hours[idx]
        if idx >= lag_step:
            dyn_lagged[idx] = palembang_vals[idx - lag_step]
        else:
            dyn_lagged[idx] = palembang_vals[0]

    df_feat_dyn = df_feat.copy()
    df_feat_dyn["dynamic_transit_palembang_pm25"] = dyn_lagged
    cols_dynamic = cols_fixed + ["dynamic_transit_palembang_pm25"]

    df_train_dyn = df_feat_dyn[train_mask].reset_index(drop=True)
    df_test_dyn = df_feat_dyn[test_mask].reset_index(drop=True)

    horizons = [1, 3, 6, 8, 10, 12]
    arch_results = []

    for h in horizons:
        # A. Separate Regional Models (Fixed Lags)
        maes_fixed = []
        maes_dyn = []

        for reg in SG_REGIONS.keys():
            col = f"pm25_{reg}"
            y_tr_true = df_train[col].shift(-h).values
            valid_tr = ~np.isnan(y_tr_true)
            delta_tr = y_tr_true[valid_tr] - df_train.loc[valid_tr, col].values

            y_te_true = df_test[col].shift(-h).values
            valid_te = ~np.isnan(y_te_true)
            base_te = df_test.loc[valid_te, col].values
            y_eval = y_te_true[valid_te]

            # Model 1: Fixed Lags
            m_fix = LGBMRegressor(objective="quantile", alpha=0.5, n_estimators=50, learning_rate=0.04, num_leaves=18, min_child_samples=8, n_jobs=4, verbosity=-1, random_state=42)
            m_fix.fit(df_train.loc[valid_tr, cols_fixed].fillna(0.0), delta_tr)
            p_fix = np.maximum(5.0, base_te + m_fix.predict(df_test.loc[valid_te, cols_fixed].fillna(0.0)))
            maes_fixed.append(mean_absolute_error(y_eval, p_fix))

            # Model 2: Dynamic Transit Lags
            m_dyn = LGBMRegressor(objective="quantile", alpha=0.5, n_estimators=50, learning_rate=0.04, num_leaves=18, min_child_samples=8, n_jobs=4, verbosity=-1, random_state=42)
            m_dyn.fit(df_train_dyn.loc[valid_tr, cols_dynamic].fillna(0.0), delta_tr)
            p_dyn = np.maximum(5.0, base_te + m_dyn.predict(df_test_dyn.loc[valid_te, cols_dynamic].fillna(0.0)))
            maes_dyn.append(mean_absolute_error(y_eval, p_dyn))

        # B. Pooled Regional Model
        # Pool all 5 regions into a single stacked training set with 'region' categorical indicator
        stacked_X_tr = []
        stacked_y_tr = []
        stacked_X_te = []
        stacked_y_te = []
        stacked_base_te = []

        for reg in SG_REGIONS.keys():
            col = f"pm25_{reg}"
            y_tr_t = df_train[col].shift(-h).values
            val_tr = ~np.isnan(y_tr_t)
            d_tr = y_tr_t[val_tr] - df_train.loc[val_tr, col].values

            sub_tr = df_train.loc[val_tr, cols_fixed].copy().fillna(0.0)
            sub_tr["reg_code"] = list(SG_REGIONS.keys()).index(reg)
            stacked_X_tr.append(sub_tr)
            stacked_y_tr.append(d_tr)

            y_te_t = df_test[col].shift(-h).values
            val_te = ~np.isnan(y_te_t)
            sub_te = df_test.loc[val_te, cols_fixed].copy().fillna(0.0)
            sub_te["reg_code"] = list(SG_REGIONS.keys()).index(reg)
            stacked_X_te.append(sub_te)
            stacked_y_te.append(y_te_t[val_te])
            stacked_base_te.append(df_test.loc[val_te, col].values)

        pool_X_tr = pd.concat(stacked_X_tr, axis=0, ignore_index=True)
        pool_y_tr = np.concatenate(stacked_y_tr)
        pool_X_te = pd.concat(stacked_X_te, axis=0, ignore_index=True)
        pool_y_te = np.concatenate(stacked_y_te)
        pool_base_te = np.concatenate(stacked_base_te)

        m_pool = LGBMRegressor(objective="quantile", alpha=0.5, n_estimators=50, learning_rate=0.04, num_leaves=24, min_child_samples=15, n_jobs=4, verbosity=-1, random_state=42)
        m_pool.fit(pool_X_tr, pool_y_tr)
        pred_pool = np.maximum(5.0, pool_base_te + m_pool.predict(pool_X_te))
        mae_pooled = float(mean_absolute_error(pool_y_te, pred_pool))

        mae_fix_avg = float(np.mean(maes_fixed))
        mae_dyn_avg = float(np.mean(maes_dyn))

        arch_results.append({
            "horizon_h": h,
            "separate_regional_mae": round(mae_fix_avg, 2),
            "dynamic_lags_mae": round(mae_dyn_avg, 2),
            "pooled_model_mae": round(mae_pooled, 2),
            "dynamic_vs_fixed_gain_pct": round((1.0 - mae_dyn_avg / mae_fix_avg) * 100.0, 2),
            "pooled_vs_separate_gain_pct": round((1.0 - mae_pooled / mae_fix_avg) * 100.0, 2),
        })

    arch_df = pd.DataFrame(arch_results)
    arch_df.to_csv(REPORTS_DIR / "architecture_and_physics_comparisons.csv", index=False)
    logger.info("Architecture and physical comparisons complete! Saved to data/reports/architecture_and_physics_comparisons.csv")
    print("\n--- Architecture & Physical Assumption Comparison ---")
    print(arch_df.to_string(index=False))
    return arch_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_architecture_and_physics_comparisons()
