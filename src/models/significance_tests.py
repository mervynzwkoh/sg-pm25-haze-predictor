"""Statistical significance testing for time series forecast comparisons.

Implements:
1. Stationary Block Bootstrap (Block length L=24h, 1000 resamples) for paired MAE loss differentials
   and 95% confidence intervals.
2. Diebold-Mariano (DM) test with rectangular/Bartlett kernel variance adjustment for h-step ahead forecasts.
"""
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from src.config import DATA_DIR

logger = logging.getLogger(__name__)
REPORTS_DIR = DATA_DIR / "reports"


def diebold_mariano_test(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
    h: int = 1,
    loss: str = "absolute"
) -> Tuple[float, float]:
    """Diebold-Mariano test for equal predictive accuracy.

    Returns: (dm_stat, p_value)
    """
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred1) & ~np.isnan(y_pred2)
    yt = y_true[mask]
    p1 = y_pred1[mask]
    p2 = y_pred2[mask]
    T = len(yt)

    if T < 10:
        return 0.0, 1.0

    if loss == "absolute":
        e1 = np.abs(yt - p1)
        e2 = np.abs(yt - p2)
    else:  # squared
        e1 = (yt - p1) ** 2
        e2 = (yt - p2) ** 2

    # Differential series: d_t = e1_t - e2_t (positive d means model 2 has smaller error)
    d = e1 - e2
    d_mean = np.mean(d)

    # Autocovariance adjustment (Newey-West / Bartlett kernel up to lag h-1)
    gamma0 = np.var(d)
    var_d = gamma0
    for k in range(1, h):
        weight = 1.0 - (k / h)
        gamma_k = np.sum((d[:-k] - d_mean) * (d[k:] - d_mean)) / T
        var_d += 2.0 * weight * gamma_k

    var_d = max(var_d, 1e-8)
    std_error = np.sqrt(var_d / T)
    dm_stat = float(d_mean / std_error)
    # Two-sided p-value from standard normal
    p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(dm_stat))))

    return dm_stat, p_val


def block_bootstrap_mae_diff(
    y_true: np.ndarray,
    y_pred_base: np.ndarray,
    y_pred_model: np.ndarray,
    block_length: int = 24,
    n_boot: int = 1000,
    seed: int = 42
) -> Dict[str, float]:
    """Moving block bootstrap on paired absolute loss differentials d_t = |e_base| - |e_model|.

    Returns mean diff, 95% CI [lower, upper], and empirical p-value (prob diff <= 0).
    """
    np.random.seed(seed)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred_base) & ~np.isnan(y_pred_model)
    yt = y_true[mask]
    pb = y_pred_base[mask]
    pm = y_pred_model[mask]
    T = len(yt)

    if T < block_length:
        return {"mean_diff": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "p_value": 1.0}

    d = np.abs(yt - pb) - np.abs(yt - pm)
    observed_mean = float(np.mean(d))

    n_blocks = int(np.ceil(T / block_length))
    boot_means = []

    for _ in range(n_boot):
        # Sample block start indices uniformly
        start_indices = np.random.randint(0, T - block_length + 1, size=n_blocks)
        sample = np.concatenate([d[s:s + block_length] for s in start_indices])[:T]
        boot_means.append(np.mean(sample))

    boot_means = np.array(boot_means)
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    # p-value: fraction of bootstrap replicates where base is equal or better than model (diff <= 0)
    p_value = float(np.mean(boot_means <= 0.0))

    return {
        "observed_mean_gain": round(observed_mean, 3),
        "ci_lower_95": round(ci_lower, 3),
        "ci_upper_95": round(ci_upper, 3),
        "p_value": round(p_value, 4),
    }


def run_statistical_significance_suite():
    """Run formal hypothesis testing on the multi-horizon predictions."""
    logger.info("Executing statistical significance suite...")

    from src.models.ablation_study import build_ablation_features
    df_raw = pd.read_csv(DATA_DIR / "raw" / "multi_year_haze_seasons_2022_2026.csv")
    df_feat, feature_groups = build_ablation_features(df_raw)

    split_date = "2026-10-01"
    train_mask = (df_feat["timestamp"] < split_date) & (df_feat["pm25_central_lag_24"].notna())
    test_mask = (df_feat["timestamp"] >= split_date)

    df_train = df_feat[train_mask].reset_index(drop=True)
    df_test = df_feat[test_mask].reset_index(drop=True)

    from lightgbm import LGBMRegressor
    from src.models.baselines import DiurnalSeasonalBaseline

    diurnal_model = DiurnalSeasonalBaseline()
    diurnal_model.fit(df_train)

    horizons = [1, 3, 6, 8, 10, 12]
    sig_results = []
    test_hours = pd.to_datetime(df_test["timestamp"]).dt.hour.values

    for h in horizons:
        for reg in ["central", "west", "south"]:
            col = f"pm25_{reg}"
            y_base_train = df_train[col].values
            y_true_train_future = df_train[col].shift(-h).values
            valid_train = ~np.isnan(y_true_train_future)
            delta_train = y_true_train_future[valid_train] - y_base_train[valid_train]

            y_base_test = df_test[col].values
            y_true_test = df_test[col].shift(-h).values
            valid_test = ~np.isnan(y_true_test)

            if np.sum(valid_test) < 20:
                continue

            y_eval = y_true_test[valid_test]
            base_eval = y_base_test[valid_test]
            test_hrs_eval = test_hours[valid_test]

            # Fit M0 and M3
            X_tr_m0 = df_train.loc[valid_train, feature_groups["M0_local_only"]].fillna(0.0)
            X_te_m0 = df_test.loc[valid_test, feature_groups["M0_local_only"]].fillna(0.0)
            m0 = LGBMRegressor(objective="quantile", alpha=0.5, n_estimators=50, learning_rate=0.04, num_leaves=18, min_child_samples=8, n_jobs=4, verbosity=-1, random_state=42)
            m0.fit(X_tr_m0, delta_train)
            pred_m0 = np.maximum(5.0, base_eval + m0.predict(X_te_m0))

            X_tr_m3 = df_train.loc[valid_train, feature_groups["M3_full_physics"]].fillna(0.0)
            X_te_m3 = df_test.loc[valid_test, feature_groups["M3_full_physics"]].fillna(0.0)
            m3 = LGBMRegressor(objective="quantile", alpha=0.5, n_estimators=50, learning_rate=0.04, num_leaves=18, min_child_samples=8, n_jobs=4, verbosity=-1, random_state=42)
            m3.fit(X_tr_m3, delta_train)
            pred_m3 = np.maximum(5.0, base_eval + m3.predict(X_te_m3))

            pred_persistence = base_eval

            # DM test: M3 vs Persistence
            dm_pers, p_pers = diebold_mariano_test(y_eval, pred_persistence, pred_m3, h=h)
            # DM test: M3 vs M0 (Local only)
            dm_m0, p_m0 = diebold_mariano_test(y_eval, pred_m0, pred_m3, h=h)

            # Block bootstrap: M3 vs Persistence
            boot_pers = block_bootstrap_mae_diff(y_eval, pred_persistence, pred_m3, block_length=24, n_boot=500)
            # Block bootstrap: M3 vs M0
            boot_m0 = block_bootstrap_mae_diff(y_eval, pred_m0, pred_m3, block_length=24, n_boot=500)

            sig_results.append({
                "horizon_h": h,
                "region": reg,
                "m3_mae": round(float(np.mean(np.abs(y_eval - pred_m3))), 2),
                "pers_mae": round(float(np.mean(np.abs(y_eval - pred_persistence))), 2),
                "m0_mae": round(float(np.mean(np.abs(y_eval - pred_m0))), 2),
                "dm_stat_vs_pers": round(dm_pers, 2),
                "dm_pval_vs_pers": round(p_pers, 4),
                "dm_stat_vs_m0": round(dm_m0, 2),
                "dm_pval_vs_m0": round(p_m0, 4),
                "boot_gain_vs_pers": boot_pers["observed_mean_gain"],
                "boot_ci95_pers": f"[{boot_pers['ci_lower_95']}, {boot_pers['ci_upper_95']}]",
                "boot_pval_pers": boot_pers["p_value"],
                "boot_gain_vs_m0": boot_m0["observed_mean_gain"],
                "boot_ci95_m0": f"[{boot_m0['ci_lower_95']}, {boot_m0['ci_upper_95']}]",
                "boot_pval_m0": boot_m0["p_value"],
            })

    sig_df = pd.DataFrame(sig_results)
    sig_df.to_csv(REPORTS_DIR / "statistical_significance_tests.csv", index=False)
    logger.info("Statistical significance suite complete! Saved to data/reports/statistical_significance_tests.csv")
    print("\n--- Statistical Significance Benchmarks (Diebold-Mariano & Block Bootstrap) ---")
    print(sig_df[["horizon_h", "region", "m3_mae", "pers_mae", "dm_stat_vs_pers", "dm_pval_vs_pers", "boot_gain_vs_pers", "boot_ci95_pers", "boot_pval_pers"]].to_string(index=False))
    return sig_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_statistical_significance_suite()
