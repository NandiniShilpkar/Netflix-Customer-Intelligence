import pandas as pd
import pytest

from src.models.train_clv_model import build_dataset
from src.services import retention_roi_simulator as roi


def make_clv_data():
    customers = pd.DataFrame({
        "customer_id": ["C1", "C2", "C3"],
        "age": [25, 30, 35],
        "gender": ["F", "M", "F"],
        "country": ["India", "India", "UK"],
        "acquisition_channel": ["Organic", "Referral", "Paid"],
        "customer_segment": ["A", "B", "A"],
        "preferred_language": ["English", "Hindi", "English"],
        "registration_date": [
            "2025-01-01", "2026-01-01", "2027-01-01"
        ],
    })

    payments = pd.DataFrame({
        "customer_id": ["C1", "C1", "C1", "C2", "C3"],
        "payment_status": [
            "Success", "Failed", "Success", "Success", "Success"
        ],
        "payment_date": [
            "2026-01-01", "2026-02-01", "2026-06-01",
            "2026-03-01", "2026-06-01"
        ],
        "amount": [100, 50, 200, 150, 300],
    })

    subscriptions = pd.DataFrame({
        "customer_id": ["C1", "C2", "C3"],
        "subscription_id": ["S1", "S2", "S3"],
        "subscription_start_date": [
            "2025-01-01", "2026-01-01", "2027-01-01"
        ],
        "subscription_end_date": [None, None, None],
        "plan_id": ["Basic", "Standard", "Premium"],
        "monthly_price": [100, 200, 300],
    })

    viewing = pd.DataFrame({
        "customer_id": ["C1", "C1", "C2"],
        "viewing_id": ["V1", "V2", "V3"],
        "viewing_date": [
            "2026-05-01", "2026-01-01", "2026-04-01"
        ],
        "watch_duration_minutes": [60, 30, 90],
        "completion_percentage": [80, 50, 90],
    })

    tickets = pd.DataFrame({
        "customer_id": ["C1", "C2"],
        "ticket_id": ["T1", "T2"],
        "ticket_date": ["2026-03-01", "2026-04-01"],
        "customer_satisfaction_score": [4, 3],
    })

    feedback = pd.DataFrame({
        "customer_id": ["C1", "C2"],
        "feedback_id": ["F1", "F2"],
        "feedback_date": ["2026-04-01", "2026-05-01"],
        "rating": [5, 3],
    })

    return {
        "customers": customers,
        "payments": payments,
        "subscriptions": subscriptions,
        "viewing_activity": viewing,
        "support_tickets": tickets,
        "customer_feedback": feedback,
    }


def test_clv_uses_only_successful_payments_before_cutoff():
    result = build_dataset(make_clv_data())

    c1 = result.loc[
        result["customer_id"] == "C1"
    ].iloc[0]

    assert c1["customer_lifetime_value"] == 100


def test_clv_excludes_customers_without_eligible_successful_payments():
    result = build_dataset(make_clv_data())

    assert set(result["customer_id"]) == {"C1", "C2"}


def test_clv_viewing_features_use_90_day_window():
    result = build_dataset(make_clv_data())

    c1 = result.loc[
        result["customer_id"] == "C1"
    ].iloc[0]

    assert c1["viewing_sessions_90d"] == 1
    assert c1["watch_minutes_90d"] == 60


def test_clv_missing_activity_counts_are_zero():
    result = build_dataset(make_clv_data())

    c2 = result.loc[
        result["customer_id"] == "C2"
    ].iloc[0]

    assert c2["viewing_sessions_90d"] == 1
    assert c2["support_ticket_count"] == 1


@pytest.fixture
def roi_test_data(monkeypatch, tmp_path):
    predictions = pd.DataFrame({
        "customer_id": ["C1", "C2", "C3", "C4"],
        "snapshot_date": pd.to_datetime([
            "2026-04-30"
        ] * 4),
        "model": ["CatBoost_B"] * 4,
        "predicted_churn_probability": [0.95, 0.90, 0.80, 0.70],
        "actual_churn": [1, 0, 1, 0],
    })

    prices = pd.DataFrame({
        "customer_id": ["C1", "C2", "C3", "C4"],
        "plan_id": ["Basic", "Basic", "Premium", "Standard"],
        "monthly_price": [100, 100, 200, 150],
    })

    monkeypatch.setattr(
        roi, "load_customer_predictions", lambda: predictions.copy()
    )
    monkeypatch.setattr(
        roi, "load_snapshot_subscription_prices", lambda date: prices.copy()
    )
    monkeypatch.setattr(roi, "OUTPUT_DIR", tmp_path)

    return tmp_path


def test_roi_simulation_calculates_expected_values(roi_test_data):
    roi.run_simulation(
        contact_percentages=[50],
        campaign_cost=10,
        retention_rate=0.5,
        months_saved=3,
    )

    result = pd.read_csv(
        roi_test_data / "retention_roi_summary.csv"
    ).iloc[0]

    assert result["customers_contacted"] == 2
    assert result["actual_churners_selected"] == 1
    assert result["expected_retained_customers"] == 0.5
    assert result["campaign_cost"] == 20
    assert result["estimated_gross_revenue_saved"] == 150
    assert result["estimated_net_benefit"] == 130
    assert result["estimated_roi_pct"] == 650


@pytest.mark.parametrize("percentage", [0, -5, 101])
def test_roi_rejects_invalid_contact_percentage(
    roi_test_data, percentage
):
    with pytest.raises(ValueError):
        roi.run_simulation(
            contact_percentages=[percentage],
            campaign_cost=10,
            retention_rate=0.5,
            months_saved=3,
        )
