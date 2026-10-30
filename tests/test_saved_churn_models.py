import joblib
import numpy as np
import pandas as pd
import pytest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/features/churn_historical_snapshots.csv"
MODEL_PATHS = [
    ROOT / "models/historical_logistic_regression.joblib",
    ROOT / "models/historical_random_forest.joblib",
]

EXCLUDED = [
    "customer_id",
    "churn_next_30_days",
    "snapshot_date",
    "outcome_window_end",
]


@pytest.fixture(scope="module")
def test_data():
    df = pd.read_csv(DATA_PATH)
    df = df[df["snapshot_date"] == "2026-05-31"].copy()

    X = df.drop(columns=EXCLUDED)
    return X


@pytest.mark.parametrize("model_path", MODEL_PATHS)
def test_saved_model_exists(model_path):
    assert model_path.exists()
    assert model_path.stat().st_size > 0


@pytest.mark.parametrize("model_path", MODEL_PATHS)
def test_saved_model_loads_and_predicts(model_path, test_data):
    model = joblib.load(model_path)

    predictions = model.predict(test_data)
    probabilities = model.predict_proba(test_data)[:, 1]

    assert len(predictions) == len(test_data)
    assert set(np.unique(predictions)).issubset({0, 1})
    assert len(probabilities) == len(test_data)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


@pytest.mark.parametrize("model_path", MODEL_PATHS)
def test_saved_model_has_two_probability_classes(model_path, test_data):
    model = joblib.load(model_path)

    probabilities = model.predict_proba(test_data)

    assert probabilities.shape == (len(test_data), 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


@pytest.mark.parametrize("model_path", MODEL_PATHS)
def test_saved_model_accepts_expected_feature_columns(model_path, test_data):
    model = joblib.load(model_path)

    expected_features = model.feature_names_in_

    assert list(test_data.columns) == list(expected_features)
