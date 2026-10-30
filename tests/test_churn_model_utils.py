import pandas as pd
import pytest

from src.models.churn_model_utils import prepare_temporal_split


TARGET = "churn_next_30_days"
EXCLUDED = [
    "customer_id",
    TARGET,
    "snapshot_date",
    "outcome_window_end",
]
TRAIN_DATES = ["2026-02-28", "2026-03-31"]
TEST_DATES = ["2026-05-31"]


def make_data():
    return pd.DataFrame({
        "customer_id": ["C1", "C2", "C3", "C4", "C5", "C6"],
        "snapshot_date": [
            "2026-02-28", "2026-02-28",
            "2026-03-31", "2026-03-31",
            "2026-05-31", "2026-05-31",
        ],
        "outcome_window_end": [
            "2026-03-30", "2026-03-30",
            "2026-04-30", "2026-04-30",
            "2026-06-30", "2026-06-30",
        ],
        TARGET: [0, 1, 0, 1, 0, 1],
        "age": [25, 30, 28, 35, 40, 32],
    })


def test_valid_temporal_split():
    train, test = prepare_temporal_split(
        make_data(), TARGET, EXCLUDED, TRAIN_DATES, TEST_DATES
    )

    assert len(train) == 4
    assert len(test) == 2
    assert train["snapshot_date"].max() < test["snapshot_date"].min()


def test_missing_required_column_is_rejected():
    data = make_data().drop(columns=["customer_id"])

    with pytest.raises(ValueError, match="Missing required columns"):
        prepare_temporal_split(
            data, TARGET, EXCLUDED, TRAIN_DATES, TEST_DATES
        )


def test_missing_target_is_rejected():
    data = make_data()
    data.loc[0, TARGET] = None

    with pytest.raises(ValueError, match="missing values"):
        prepare_temporal_split(
            data, TARGET, EXCLUDED, TRAIN_DATES, TEST_DATES
        )


def test_invalid_target_values_are_rejected():
    data = make_data()
    data.loc[0, TARGET] = 2

    with pytest.raises(ValueError, match="only 0 and 1"):
        prepare_temporal_split(
            data, TARGET, EXCLUDED, TRAIN_DATES, TEST_DATES
        )


def test_duplicate_customer_snapshot_is_rejected():
    data = make_data()
    data.loc[1, "customer_id"] = "C1"

    with pytest.raises(ValueError, match="Duplicate customer-snapshot"):
        prepare_temporal_split(
            data, TARGET, EXCLUDED, TRAIN_DATES, TEST_DATES
        )


def test_empty_test_period_is_rejected():
    with pytest.raises(ValueError, match="dataset is empty"):
        prepare_temporal_split(
            make_data(), TARGET, EXCLUDED, TRAIN_DATES, ["2026-06-30"]
        )


def test_overlapping_outcome_window_is_rejected():
    data = make_data()
    data.loc[2:3, "outcome_window_end"] = "2026-05-31"

    with pytest.raises(ValueError, match="overlap the test period"):
        prepare_temporal_split(
            data, TARGET, EXCLUDED, TRAIN_DATES, TEST_DATES
        )


def test_single_class_in_test_period_is_rejected():
    data = make_data()
    data.loc[5, TARGET] = 0

    with pytest.raises(ValueError, match="both target classes"):
        prepare_temporal_split(
            data, TARGET, EXCLUDED, TRAIN_DATES, TEST_DATES
        )
