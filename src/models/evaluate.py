"""Evaluation metrics for point and probabilistic time series forecasting."""
import numpy as np
from typing import Dict, Any


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate point forecast error metrics (MAE, RMSE, MAPE)."""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt = np.asarray(y_true)[mask]
    yp = np.asarray(y_pred)[mask]

    if len(yt) == 0:
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0}

    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    # Safe MAPE with floor at 1.0 µg/m³ to prevent division by zero
    mape = float(np.mean(np.abs((yt - yp) / np.maximum(yt, 1.0))) * 100.0)

    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 2),
    }


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Calculate the quantile pinball loss for quantile alpha."""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt = np.asarray(y_true)[mask]
    yp = np.asarray(y_pred)[mask]

    if len(yt) == 0:
        return 0.0

    diff = yt - yp
    loss = np.maximum(alpha * diff, (alpha - 1.0) * diff)
    return float(np.mean(loss))


def interval_coverage(y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray) -> float:
    """Calculate the Prediction Interval Coverage Probability (PICP).

    Returns percentage of ground truth observations inside [y_lower, y_upper].
    """
    mask = ~np.isnan(y_true) & ~np.isnan(y_lower) & ~np.isnan(y_upper)
    yt = np.asarray(y_true)[mask]
    yl = np.asarray(y_lower)[mask]
    yu = np.asarray(y_upper)[mask]

    if len(yt) == 0:
        return 0.0

    covered = (yt >= yl) & (yt <= yu)
    return float(np.mean(covered) * 100.0)
