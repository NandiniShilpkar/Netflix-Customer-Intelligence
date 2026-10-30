import pandas as pd
import pytest

from src.models import segment_customers as segmentation


class FakeConnection:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class FakeEngine:
    def connect(self):
        return FakeConnection()


def test_load_customer_features_returns_valid_records(monkeypatch):
    expected = pd.DataFrame({
        "customer_id": ["C1", "C2", "C3"],
        "age": [25, 30, 35],
        "country": ["India", "USA", "UK"],
    })

    captured = {}

    def fake_read_sql(query, connection, params):
        captured["params"] = params
        return expected.copy()

    monkeypatch.setattr(segmentation.pd, "read_sql", fake_read_sql)

    result = segmentation.load_customer_features(FakeEngine())

    assert len(result) == 3
    assert result["customer_id"].is_unique
    assert captured["params"]["cutoff"] == "2026-06-30"


def test_load_customer_features_rejects_empty_data(monkeypatch):
    monkeypatch.setattr(
        segmentation.pd,
        "read_sql",
        lambda *args, **kwargs: pd.DataFrame()
    )

    with pytest.raises(ValueError, match="No customer records"):
        segmentation.load_customer_features(FakeEngine())


def test_load_customer_features_rejects_duplicate_ids(monkeypatch):
    duplicate_data = pd.DataFrame({
        "customer_id": ["C1", "C1", "C2"],
        "age": [25, 25, 30],
    })

    monkeypatch.setattr(
        segmentation.pd,
        "read_sql",
        lambda *args, **kwargs: duplicate_data.copy()
    )

    with pytest.raises(ValueError, match="duplicate customer IDs"):
        segmentation.load_customer_features(FakeEngine())


def test_segmentation_cutoff_is_fixed():
    assert segmentation.CUTOFF_DATE == "2026-06-30"


def test_segmentation_cluster_configuration():
    assert segmentation.MIN_CLUSTERS == 2
    assert segmentation.MAX_CLUSTERS == 7
    assert segmentation.RANDOM_STATE == 42
