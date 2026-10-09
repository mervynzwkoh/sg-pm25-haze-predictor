"""Model training and walk-forward validation pipeline.

Trains multi-horizon LightGBM Quantile Regressors (p10, p50, p90)
for horizons h=1..12 across all 5 Singapore planning regions.
Compares against Persistence and Ridge baselines.
"""
import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

from src.config import SG_REGIONS, FORECAST_HORIZONS, QUANTILES, DATA_DIR
from src.features.build_features import (
    add_time_features,
    add_wind_features,
    add_autoregressive_features,
    add_transboundary_advection_features,
    prepare_training_dataset,
)
from src.models.evaluate import calculate_metrics, interval_coverage, pinball_loss

logger = logging.getLogger(__name__)

MODELS_DIR = DATA_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class MultiHorizonForecaster:
    """Multi-horizon multi-region probabilistic forecaster."""

    def __init__(self, horizons: List[int] = FORECAST_HORIZONS, quantiles: List[float] = QUANTILES):
        self.horizons = horizons
        self.quantiles = quantiles
        self.models: Dict[str, Dict[int, Dict[float, LGBMRegressor]]] = {}
        self.feature_cols: List[str] = []

    def train(
        self,
        X_train: pd.DataFrame,
        targets_train: Dict[str, Dict[int, pd.Series]],
        feature_cols: List[str]
    ):
        """Train LightGBM quantile regression models for each region, horizon, and quantile."""
        self.feature_cols = feature_cols
        X = X_train[feature_cols].ffill().fillna(0.0)

        for reg in SG_REGIONS.keys():
            self.models[reg] = {}
            for h in self.horizons:
                self.models[reg][h] = {}
                y = targets_train[reg][h]
                valid_mask = ~y.isna()
                X_h = X[valid_mask]
                y_h = y[valid_mask]

                for q in self.quantiles:
                    # LightGBM quantile objective
                    model = LGBMRegressor(
                        objective="quantile",
                        alpha=q,
                        n_estimators=75,
                        learning_rate=0.08,
                        num_leaves=20,
                        min_child_samples=5,
                        verbosity=-1,
                        random_state=42
                    )
                    model.fit(X_h, y_h)
                    self.models[reg][h][q] = model

    def predict_horizon(
        self,
        X_row: pd.DataFrame,
        region: str,
        horizon: int,
        current_val: Optional[float] = None
    ) -> Dict[str, float]:
        """Predict p10, p50 (median), and p90 anchored to the current observed PM2.5 reading."""
        X = X_row[self.feature_cols].fillna(0.0)

        # Base reading anchor
        if current_val is not None:
            base = float(current_val)
        elif f"pm25_{region}" in X_row.columns:
            base = float(X_row[f"pm25_{region}"].iloc[0])
        else:
            base = 35.0

        delta_10 = float(self.models[region][horizon][0.10].predict(X)[0])
        delta_50 = float(self.models[region][horizon][0.50].predict(X)[0])
        delta_90 = float(self.models[region][horizon][0.90].predict(X)[0])

        p50 = max(5.0, base + delta_50)
        p10 = max(5.0, min(p50, base + delta_10))
        p90 = max(p50, base + delta_90)

        return {"p10": round(p10, 1), "p50": round(p50, 1), "p90": round(p90, 1)}

    def save(self, filepath: Path):
        """Serialize trained models to disk."""
        with open(filepath, "wb") as f:
            pickle.dump({"models": self.models, "feature_cols": self.feature_cols}, f)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "MultiHorizonForecaster":
        """Load serialized models from disk."""
        forecaster = cls()
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            forecaster.models = data["models"]
            forecaster.feature_cols = data["feature_cols"]
        return forecaster


def evaluate_forecast_horizons(
    forecaster: MultiHorizonForecaster,
    X_test: pd.DataFrame,
    targets_test: Dict[str, Dict[int, pd.Series]],
    df_test: pd.DataFrame
) -> pd.DataFrame:
    """Benchmark Adaptive LightGBM against Persistence baseline across all lead times h=1..12."""
    results = []

    for h in forecaster.horizons:
        for reg in SG_REGIONS.keys():
            col = f"pm25_{reg}"
            # True level is base y_t + true delta
            delta_true = targets_test[reg][h].values
            valid_mask = ~np.isnan(delta_true)

            if np.sum(valid_mask) == 0:
                continue

            X_valid = X_test[valid_mask][forecaster.feature_cols].fillna(0.0)
            y_base = df_test.loc[valid_mask, col].values
            y_true = y_base + delta_true[valid_mask]

            # Adaptive LightGBM delta predictions
            pred_delta_50 = forecaster.models[reg][h][0.50].predict(X_valid)
            pred_p50 = np.maximum(5.0, y_base + pred_delta_50)

            pred_delta_10 = forecaster.models[reg][h][0.10].predict(X_valid)
            pred_p10 = np.maximum(5.0, np.minimum(pred_p50, y_base + pred_delta_10))

            pred_delta_90 = forecaster.models[reg][h][0.90].predict(X_valid)
            pred_p90 = np.maximum(pred_p50, y_base + pred_delta_90)

            lgbm_metrics = calculate_metrics(y_true, pred_p50)
            picp = interval_coverage(y_true, pred_p10, pred_p90)

            # Persistence Baseline: y_{t+h} predicted as y_t (last observed value)
            persistence_metrics = calculate_metrics(y_true, y_base)

            results.append({
                "horizon_h": h,
                "region": reg,
                "adaptive_mae": lgbm_metrics["mae"],
                "adaptive_rmse": lgbm_metrics["rmse"],
                "coverage_80pct": round(picp, 1),
                "persistence_mae": persistence_metrics["mae"],
                "persistence_rmse": persistence_metrics["rmse"],
                "skill_gain_pct": round((1.0 - lgbm_metrics["mae"] / max(persistence_metrics["mae"], 0.01)) * 100.0, 1),
            })

    return pd.DataFrame(results)
