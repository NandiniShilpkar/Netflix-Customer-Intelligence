import joblib
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/customer_segmentation.joblib"


@pytest.fixture(scope="module")
def artifact():
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def sample_features(artifact):
    numeric = artifact["numeric_columns"]
    categorical = artifact["categorical_columns"]

    data = {}

    for column in numeric:
        data[column] = [0.0, 1.0, 2.0]

    for column in categorical:
        data[column] = ["test_category", "unknown_category", "test_category"]

    return pd.DataFrame(data)[artifact["feature_columns"]]


def test_segmentation_artifact_exists():
    assert MODEL_PATH.exists()
    assert MODEL_PATH.stat().st_size > 0


def test_segmentation_artifact_configuration(artifact):
    assert artifact["selected_k"] == artifact["model"].n_clusters
    assert artifact["selected_k"] >= 2
    assert len(artifact["feature_columns"]) == 19


def test_preprocessor_transforms_customer_features(artifact, sample_features):
    transformed = artifact["preprocessor"].transform(sample_features)

    assert transformed.shape[0] == len(sample_features)
    assert transformed.shape[1] > 0

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    assert np.isfinite(transformed).all()


def test_model_assigns_valid_segments(artifact, sample_features):
    transformed = artifact["preprocessor"].transform(sample_features)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    predictions = artifact["model"].predict(transformed)

    assert len(predictions) == len(sample_features)
    assert set(np.unique(predictions)).issubset(
        set(range(artifact["selected_k"]))
    )


def test_segmentation_predictions_are_deterministic(artifact, sample_features):
    transformed = artifact["preprocessor"].transform(sample_features)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    first = artifact["model"].predict(transformed)
    second = artifact["model"].predict(transformed)

    assert np.array_equal(first, second)


def test_cluster_centers_match_transformed_feature_count(artifact):
    assert artifact["model"].cluster_centers_.shape[0] == artifact["selected_k"]
    assert artifact["model"].cluster_centers_.shape[1] > 0
