"""Unit tests for model training, metrics, and confidence scoring."""
import pytest
import numpy as np
import pandas as pd

from src.models.evaluate import calculate_metrics, pinball_loss, interval_coverage
from src.models.predict import compute_confidence_score, get_health_band
from src.models.train import MultiHorizonForecaster


def test_metrics_calculation():
    y_true = np.array([20.0, 30.0, 50.0])
    y_pred = np.array([22.0, 28.0, 55.0])
    m = calculate_metrics(y_true, y_pred)

    assert pytest.approx(m["mae"], 0.1) == 3.0
    assert m["rmse"] > 0
    assert m["mape"] > 0


def test_pinball_loss():
    y_true = np.array([50.0])
    y_pred = np.array([40.0])
    # Underprediction (diff = 10 > 0) with alpha=0.9 -> loss = 0.9 * 10 = 9.0
    loss_high = pinball_loss(y_true, y_pred, 0.90)
    assert pytest.approx(loss_high, 0.01) == 9.0


def test_interval_coverage():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_lower = np.array([5.0, 15.0, 25.0, 45.0])  # 3 of 4 covered
    y_upper = np.array([15.0, 25.0, 35.0, 55.0])
    cov = interval_coverage(y_true, y_lower, y_upper)
    assert cov == 75.0


def test_confidence_score():
    # Narrow interval -> higher confidence
    c_high = compute_confidence_score(p10=30.0, p50=35.0, p90=40.0, horizon_h=1)
    # Wide interval -> lower confidence
    c_low = compute_confidence_score(p10=10.0, p50=50.0, p90=120.0, horizon_h=12)

    assert c_high > c_low
    assert 20.0 <= c_high <= 100.0
    assert 20.0 <= c_low <= 100.0


def test_health_band_mapping():
    band_safe = get_health_band(35.0)
    assert band_safe["band"] == "Normal"
    assert band_safe["badge"] == "Safe"

    band_elev = get_health_band(85.0)
    assert band_elev["band"] == "Elevated"
    assert band_elev["badge"] == "Caution"

    band_unhealthy = get_health_band(180.0)
    assert band_unhealthy["band"] == "Unhealthy"


def test_multi_horizon_forecaster_fit_predict():
    # Generate synthetic training set
    np.random.seed(42)
    n = 60
    X = pd.DataFrame({
        "feat1": np.random.randn(n),
        "feat2": np.random.randn(n),
    })
    feature_cols = ["feat1", "feat2"]

    targets = {}
    for r in ["north", "south", "east", "west", "central"]:
        targets[r] = {
            1: pd.Series(np.random.uniform(20, 100, n)),
            2: pd.Series(np.random.uniform(20, 100, n)),
        }

    forecaster = MultiHorizonForecaster(horizons=[1, 2], quantiles=[0.10, 0.50, 0.90])
    forecaster.train(X, targets, feature_cols)

    pred = forecaster.predict_horizon(X.iloc[[0]], "north", 1)
    assert "p10" in pred and "p50" in pred and "p90" in pred
    assert pred["p10"] <= pred["p50"] <= pred["p90"]
