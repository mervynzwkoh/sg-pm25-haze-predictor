"""Strong comparative baseline models for time-series PM2.5 forecasting.

Implements:
1. Persistence: y_{t+h} = y_t
2. Diurnal Seasonal Persistence / Climatology: y_{t+h} = y_t + Delta_{mean}(h, hour)
3. Regularized Autoregressive (Ridge AR): Linear multi-horizon regressor on lag matrix
4. Standard LightGBM (No Physics): Non-physics GBDT using local lags and local temperature/RH only
"""
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor

from src.config import SG_REGIONS, FORECAST_HORIZONS


class PersistenceBaseline:
    """Naive persistence forecaster: y_{t+h} = y_t."""

    def predict(self, current_val: float, horizon: int) -> float:
        return float(current_val)


class DiurnalSeasonalBaseline:
    """Diurnal climatology: predicts y_t + mean diurnal delta for (hour, horizon)."""

    def __init__(self, horizons: List[int] = FORECAST_HORIZONS):
        self.horizons = horizons
        # Map: (region, horizon, hour) -> mean_delta
        self.climatology: Dict[str, Dict[int, Dict[int, float]]] = {}

    def fit(self, df: pd.DataFrame):
        df = df.copy()
        ts = pd.to_datetime(df["timestamp"])
        hours = ts.dt.hour

        for reg in SG_REGIONS.keys():
            col = f"pm25_{reg}"
            if col not in df.columns:
                continue
            self.climatology[reg] = {}
            for h in self.horizons:
                self.climatology[reg][h] = {}
                delta = df[col].shift(-h) - df[col]
                temp_df = pd.DataFrame({"hour": hours, "delta": delta}).dropna()
                grouped = temp_df.groupby("hour")["delta"].mean()
                for hr in range(24):
                    self.climatology[reg][h][hr] = float(grouped.get(hr, 0.0))

    def predict(self, current_val: float, region: str, horizon: int, hour: int) -> float:
        delta = self.climatology.get(region, {}).get(horizon, {}).get(hour, 0.0)
        return float(max(5.0, current_val + delta))


class RidgeARBaseline:
    """Regularized Autoregressive Linear Model (Ridge AR)."""

    def __init__(self, horizons: List[int] = FORECAST_HORIZONS, alpha: float = 1.0):
        self.horizons = horizons
        self.alpha = alpha
        self.models: Dict[str, Dict[int, Ridge]] = {}
        self.feature_cols: List[str] = []

    def fit(self, X_train: pd.DataFrame, targets_train: Dict[str, Dict[int, pd.Series]], lag_cols: List[str]):
        self.feature_cols = lag_cols
        X = X_train[lag_cols].fillna(0.0)

        for reg in SG_REGIONS.keys():
            self.models[reg] = {}
            for h in self.horizons:
                y = targets_train[reg][h]
                valid = ~y.isna()
                m = Ridge(alpha=self.alpha)
                m.fit(X[valid], y[valid])
                self.models[reg][h] = m

    def predict(self, X_row: pd.DataFrame, current_val: float, region: str, horizon: int) -> float:
        X = X_row[self.feature_cols].fillna(0.0)
        delta = float(self.models[region][horizon].predict(X)[0])
        return float(max(5.0, current_val + delta))


class StandardLGBMBaseline:
    """Standard LightGBM regressor using ONLY local autoregressive lags and local weather (no physics)."""

    def __init__(self, horizons: List[int] = FORECAST_HORIZONS):
        self.horizons = horizons
        self.models: Dict[str, Dict[int, LGBMRegressor]] = {}
        self.feature_cols: List[str] = []

    def fit(self, X_train: pd.DataFrame, targets_train: Dict[str, Dict[int, pd.Series]], local_feature_cols: List[str]):
        self.feature_cols = local_feature_cols
        X = X_train[local_feature_cols].fillna(0.0)

        for reg in SG_REGIONS.keys():
            self.models[reg] = {}
            for h in self.horizons:
                y = targets_train[reg][h]
                valid = ~y.isna()
                m = LGBMRegressor(
                    n_estimators=75,
                    learning_rate=0.04,
                    num_leaves=18,
                    min_child_samples=8,
                    verbosity=-1,
                    random_state=42
                )
                m.fit(X[valid], y[valid])
                self.models[reg][h] = m

    def predict(self, X_row: pd.DataFrame, current_val: float, region: str, horizon: int) -> float:
        X = X_row[self.feature_cols].fillna(0.0)
        delta = float(self.models[region][horizon].predict(X)[0])
        return float(max(5.0, current_val + delta))
