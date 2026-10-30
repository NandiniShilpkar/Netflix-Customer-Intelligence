import json
import joblib
import numpy as np
import pandas as pd
import pytest

from pathlib import Path
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/features/churn_historical_snapshots.csv"
REPORT_PATH = ROOT / "reports/historical_churn_metrics.json"

MODELS = {
    "logistic_regression": ROOT / "models/historical_logistic_regression.joblib",
    "random_forest": ROOT / "models/historical_random_forest.joblib",
}

TARGET = "churn_next_30_days"
EXCLUDED = [
    "customer_id",
    TARGET,
    "snapshot_date",
    "outcome_window_end",
]


@pytest.fixture(scope="module")
def evaluation_data():
    df = pd.read_csv(DATA_PATH)
    df = df[df["snapshot_date"] == "2026-05-31"].copy()

    X = df.drop(columns=EXCLUDED)
    y = df[TARGET].astype(int)

    return X, y


@pytest.fixture(scope="module")
def evaluation_report():
    with open(REPORT_PATH, encoding="utf-8") as file:
        return json.load(file)


@pytest.mark.parametrize("model_name", MODELS.keys())
def test_saved_metrics_match_model_predictions(
    model_name, evaluation_data, evaluation_report
):
    X, y = evaluation_data
    model = joblib.load(MODELS[model_name])

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    actual = evaluation_report["metrics"][model_name]

    assert precision_score(y, predictions, zero_division=0) == pytest.approx(actual["precision"])
    assert recall_score(y, predictions, zero_division=0) == pytest.approx(actual["recall"])
    assert f1_score(y, predictions, zero_division=0) == pytest.approx(actual["f1"])
    assert roc_auc_score(y, probabilities) == pytest.approx(actual["roc_auc"])
    assert average_precision_score(y, probabilities) == pytest.approx(actual["average_precision"])

    assert confusion_matrix(y, predictions, labels=[0, 1]).tolist() == actual["confusion_matrix"]


def test_evaluation_report_snapshot_metadata(evaluation_report):
    assert evaluation_report["training_snapshots"] == [
        "2026-02-28", "2026-03-31"
    ]
    assert evaluation_report["testing_snapshots"] == ["2026-05-31"]
    assert evaluation_report["training_rows"] == 13252
    assert evaluation_report["testing_rows"] == 6260


def test_test_dataset_churn_prevalence(evaluation_data):
    _, y = evaluation_data

    assert len(y) == 6260
    assert int(y.sum()) == 251
    assert y.mean() == pytest.approx(251 / 6260)


@pytest.mark.parametrize("model_name", MODELS.keys())
def test_evaluation_metrics_are_valid(model_name, evaluation_report):
    metrics = evaluation_report["metrics"][model_name]

    for name in [
        "precision", "recall", "f1", "roc_auc", "average_precision"
    ]:
        assert 0 <= metrics[name] <= 1

    assert np.isfinite(list(metrics.values())[:5]).all()
