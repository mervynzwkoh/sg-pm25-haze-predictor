"""Conformalized Quantile Regression (CQR), Threshold Exceedance Probabilities,

and Regime-Stratified Evaluation.

Implements:
1. Conformalized Quantile Regression (Romano et al., 2019) using an independent calibration fold
   to guarantee ~80% empirical coverage.
2. Validated Exceedance Probabilities P(PM2.5 > 55) and P(PM2.5 > 150) using CDF interpolation.
3. Regime-Stratified Evaluation: Normal, Pollution Onset, Severe Haze, Post-Rainfall Clearance.
"""
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.interpolate import interp1d

from src.config import SG_REGIONS, FORECAST_HORIZONS, DATA_DIR

logger = logging.getLogger(__name__)
REPORTS_DIR = DATA_DIR / "reports"


class ConformalQuantileCalibrator:
    """Computes non-conformity scores and conformal quantile corrections."""

    def __init__(self, nominal_coverage: float = 0.80):
        self.nominal_coverage = nominal_coverage
        self.alpha = 1.0 - nominal_coverage  # 0.20 for 80% coverage
        # Map: (region, horizon) -> Q_correction
        self.q_corrections: Dict[Tuple[str, int], float] = {}

    def calibrate(
        self,
        y_cal_true: np.ndarray,
        p10_cal: np.ndarray,
        p90_cal: np.ndarray,
        region: str,
        horizon: int
    ) -> float:
        """Compute conformal correction Q on independent calibration fold."""
        mask = ~np.isnan(y_cal_true) & ~np.isnan(p10_cal) & ~np.isnan(p90_cal)
        yt = y_cal_true[mask]
        l = p10_cal[mask]
        u = p90_cal[mask]
        n = len(yt)

        if n < 10:
            self.q_corrections[(region, horizon)] = 0.0
            return 0.0

        # Non-conformity scores: E_i = max(l_i - y_i, y_i - u_i)
        scores = np.maximum(l - yt, yt - u)
        level = min(1.0, np.ceil((n + 1) * self.nominal_coverage) / n)
        q_val = float(np.quantile(scores, level))
        self.q_corrections[(region, horizon)] = q_val
        return q_val

    def transform(
        self,
        p10_pred: np.ndarray,
        p90_pred: np.ndarray,
        region: str,
        horizon: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply conformal correction to test prediction intervals."""
        q = self.q_corrections.get((region, horizon), 0.0)
        p10_cqr = np.maximum(5.0, p10_pred - q)
        p90_cqr = np.maximum(p10_cqr + 1.0, p90_pred + q)
        return p10_cqr, p90_cqr


def estimate_exceedance_probability(
    p10: float,
    p50: float,
    p90: float,
    threshold: float
) -> float:
    """Estimate P(PM2.5 > threshold) using piecewise linear quantile CDF interpolation."""
    # Empirical quantiles: (5.0 -> 0.0), (p10 -> 0.10), (p50 -> 0.50), (p90 -> 0.90), (p90 + spread -> 0.99)
    spread = max(10.0, p90 - p50)
    p_floor = max(0.0, min(5.0, p10 - spread))
    p_max = p90 + (2.0 * spread)

    x_vals = [p_floor, p10, p50, p90, p_max]
    y_vals = [0.01, 0.10, 0.50, 0.90, 0.99]

    # Monotonic sort
    for i in range(1, len(x_vals)):
        if x_vals[i] <= x_vals[i - 1]:
            x_vals[i] = x_vals[i - 1] + 0.1

    if threshold <= x_vals[0]:
        return 99.0
    if threshold >= x_vals[-1]:
        return 1.0

    cdf_val = float(np.interp(threshold, x_vals, y_vals))
    prob_exceed = (1.0 - cdf_val) * 100.0
    return round(float(np.clip(prob_exceed, 1.0, 99.0)), 1)


def run_conformal_and_regime_evaluation():
    """Execute complete CQR calibration and regime-stratified evaluation."""
    logger.info("Running Conformalized Quantile Regression and Regime-Stratified Evaluation...")

    from src.models.ablation_study import build_ablation_features
    df_raw = pd.read_csv(DATA_DIR / "raw" / "multi_year_haze_seasons_2022_2026.csv")
    df_feat, feature_groups = build_ablation_features(df_raw)

    f_cols = feature_groups["M3_full_physics"]

    # Temporal partitioning:
    # 1. Training fold: 2022 to 2024
    # 2. Calibration fold: 2025 season
    # 3. Test holdout fold: October 2026 severe haze surge
    ts = pd.to_datetime(df_feat["timestamp"])
    train_mask = (ts < "2025-01-01") & df_feat["pm25_central_lag_24"].notna()
    cal_mask = (ts >= "2025-08-01") & (ts < "2026-01-01")
    test_mask = (ts >= "2026-10-01")

    df_train = df_feat[train_mask].reset_index(drop=True)
    df_cal = df_feat[cal_mask].reset_index(drop=True)
    df_test = df_feat[test_mask].reset_index(drop=True)

    logger.info(f"Train samples: {len(df_train)} | Cal samples: {len(df_cal)} | Test samples: {len(df_test)}")

    calibrator = ConformalQuantileCalibrator(nominal_coverage=0.80)
    horizons = [1, 2, 3, 4, 6, 8, 10, 12]

    cqr_benchmarks = []
    regime_benchmarks = []

    for h in horizons:
        for reg in SG_REGIONS.keys():
            col = f"pm25_{reg}"

            # 1. Targets
            y_tr_base = df_train[col].values
            y_tr_true = df_train[col].shift(-h).values
            val_tr = ~np.isnan(y_tr_true)
            delta_tr = y_tr_true[val_tr] - y_tr_base[val_tr]

            y_cal_base = df_cal[col].values
            y_cal_true = df_cal[col].shift(-h).values
            val_cal = ~np.isnan(y_cal_true)

            y_te_base = df_test[col].values
            y_te_true = df_test[col].shift(-h).values
            val_te = ~np.isnan(y_te_true)

            if np.sum(val_cal) < 10 or np.sum(val_te) < 10:
                continue

            X_tr = df_train.loc[val_tr, f_cols].fillna(0.0)
            X_cal = df_cal.loc[val_cal, f_cols].fillna(0.0)
            X_te = df_test.loc[val_te, f_cols].fillna(0.0)

            # Fit 3 quantiles: p10, p50, p90
            m10 = LGBMRegressor(objective="quantile", alpha=0.10, n_estimators=50, learning_rate=0.04, num_leaves=18, min_child_samples=8, n_jobs=4, verbosity=-1, random_state=42)
            m50 = LGBMRegressor(objective="quantile", alpha=0.50, n_estimators=50, learning_rate=0.04, num_leaves=18, min_child_samples=8, n_jobs=4, verbosity=-1, random_state=42)
            m90 = LGBMRegressor(objective="quantile", alpha=0.90, n_estimators=50, learning_rate=0.04, num_leaves=18, min_child_samples=8, n_jobs=4, verbosity=-1, random_state=42)

            m10.fit(X_tr, delta_tr)
            m50.fit(X_tr, delta_tr)
            m90.fit(X_tr, delta_tr)

            # Predictions on calibration fold
            p10_cal = np.maximum(5.0, y_cal_base[val_cal] + m10.predict(X_cal))
            p50_cal = np.maximum(5.0, y_cal_base[val_cal] + m50.predict(X_cal))
            p90_cal = np.maximum(p50_cal, y_cal_base[val_cal] + m90.predict(X_cal))
            y_cal_eval = y_cal_true[val_cal]

            # Fit Conformal correction
            q_corr = calibrator.calibrate(y_cal_eval, p10_cal, p90_cal, reg, h)

            # Predictions on test holdout fold
            p10_te_raw = np.maximum(5.0, y_te_base[val_te] + m10.predict(X_te))
            p50_te = np.maximum(5.0, y_te_base[val_te] + m50.predict(X_te))
            p90_te_raw = np.maximum(p50_te, y_te_base[val_te] + m90.predict(X_te))

            # Apply Conformal Calibration
            p10_te_cqr, p90_te_cqr = calibrator.transform(p10_te_raw, p90_te_raw, reg, h)

            y_te_eval = y_te_true[val_te]

            # Compute Uncalibrated vs Conformalized Coverage
            raw_covered = np.mean((y_te_eval >= p10_te_raw) & (y_te_eval <= p90_te_raw)) * 100.0
            cqr_covered = np.mean((y_te_eval >= p10_te_cqr) & (y_te_eval <= p90_te_cqr)) * 100.0
            raw_width = float(np.mean(p90_te_raw - p10_te_raw))
            cqr_width = float(np.mean(p90_te_cqr - p10_te_cqr))

            cqr_benchmarks.append({
                "horizon_h": h,
                "region": reg,
                "q_conformal_correction": round(q_corr, 2),
                "raw_coverage_pct": round(raw_covered, 1),
                "cqr_coverage_pct": round(cqr_covered, 1),
                "raw_interval_width": round(raw_width, 2),
                "cqr_interval_width": round(cqr_width, 2),
            })

            # Regime Stratification on test set
            # 1. Normal Regime (actual <= 55)
            # 2. Pollution Onset (actual > 55 and diff_3h > 15)
            # 3. Severe Haze Peak (actual > 100)
            # 4. Post-Rainfall Clearance (rain_3h > 2.0 mm)
            diff3h = df_test.loc[val_te, f"pm25_{reg}_diff_3h"].values if f"pm25_{reg}_diff_3h" in df_test.columns else np.zeros_like(y_te_eval)
            rain3h = df_test.loc[val_te, "rain_scavenging_3h"].values if "rain_scavenging_3h" in df_test.columns else np.zeros_like(y_te_eval)

            regimes = {
                "Normal (<=55)": (y_te_eval <= 55.0),
                "Pollution Onset": (y_te_eval > 55.0) & (diff3h > 10.0),
                "Severe Haze (>100)": (y_te_eval > 100.0),
                "Post-Rainfall Washout": (rain3h > 1.0),
            }

            for reg_name, r_mask in regimes.items():
                if np.sum(r_mask) >= 5:
                    sub_y = y_te_eval[r_mask]
                    sub_p50 = p50_te[r_mask]
                    sub_p10_cqr = p10_te_cqr[r_mask]
                    sub_p90_cqr = p90_te_cqr[r_mask]
                    sub_base = y_te_base[val_te][r_mask]

                    mae_m = float(np.mean(np.abs(sub_y - sub_p50)))
                    mae_base = float(np.mean(np.abs(sub_y - sub_base)))
                    cov_cqr = float(np.mean((sub_y >= sub_p10_cqr) & (sub_y <= sub_p90_cqr)) * 100.0)

                    regime_benchmarks.append({
                        "horizon_h": h,
                        "region": reg,
                        "regime": reg_name,
                        "sample_count": int(np.sum(r_mask)),
                        "model_mae": round(mae_m, 2),
                        "persistence_mae": round(mae_base, 2),
                        "skill_gain_pct": round((1.0 - mae_m / max(0.01, mae_base)) * 100.0, 2),
                        "cqr_coverage_pct": round(cov_cqr, 1),
                    })

    cqr_df = pd.DataFrame(cqr_benchmarks)
    cqr_df.to_csv(REPORTS_DIR / "conformal_coverage_calibration.csv", index=False)

    reg_df = pd.DataFrame(regime_benchmarks)
    reg_df.to_csv(REPORTS_DIR / "regime_stratified_evaluation.csv", index=False)

    logger.info("=== CONFORMAL CALIBRATION & REGIME EVALUATION COMPLETE ===")

    print("\n--- Conformalized Quantile Regression Coverage (Target: 80%) ---")
    cqr_summary = cqr_df.groupby("horizon_h")[["raw_coverage_pct", "cqr_coverage_pct", "raw_interval_width", "cqr_interval_width"]].mean()
    print(cqr_summary.to_string())

    print("\n--- Regime-Stratified Performance (Averaged across regions & horizons) ---")
    reg_summary = reg_df.groupby("regime")[["sample_count", "model_mae", "persistence_mae", "skill_gain_pct", "cqr_coverage_pct"]].mean()
    print(reg_summary.to_string())

    return {"cqr_df": cqr_df, "regime_df": reg_df}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_conformal_and_regime_evaluation()
