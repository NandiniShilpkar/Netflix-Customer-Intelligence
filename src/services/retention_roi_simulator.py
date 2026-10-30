import argparse
from pathlib import Path

import pandas as pd


PREDICTIONS_FILE = Path(
    "reports/threshold_analysis/april_customer_predictions.csv"
)
SUBSCRIPTIONS_FILE = Path("data/processed/subscriptions.csv")
OUTPUT_DIR = Path("reports/retention_roi")


def load_customer_predictions():
    df = pd.read_csv(PREDICTIONS_FILE)

    # Use the current leading business-ranking candidate.
    df = df[df["model"] == "CatBoost_B"].copy()

    if df.empty:
        raise ValueError("No CatBoost_B predictions found.")

    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    df = df.sort_values(
        "predicted_churn_probability", ascending=False
    ).reset_index(drop=True)

    return df


def load_snapshot_subscription_prices(snapshot_date):
    subscriptions = pd.read_csv(SUBSCRIPTIONS_FILE)

    subscriptions["subscription_start_date"] = pd.to_datetime(
        subscriptions["subscription_start_date"], errors="coerce"
    )
    subscriptions["subscription_end_date"] = pd.to_datetime(
        subscriptions["subscription_end_date"], errors="coerce"
    )
    subscriptions["cancellation_date"] = pd.to_datetime(
        subscriptions["cancellation_date"], errors="coerce"
    )

    subscriptions["monthly_price"] = pd.to_numeric(
        subscriptions["monthly_price"], errors="coerce"
    )

    # Reconstruct which subscription was valid on the snapshot date.
    valid = subscriptions[
        (subscriptions["subscription_start_date"] <= snapshot_date)
        & (
            subscriptions["subscription_end_date"].isna()
            | (subscriptions["subscription_end_date"] >= snapshot_date)
        )
        & (
            subscriptions["cancellation_date"].isna()
            | (subscriptions["cancellation_date"] > snapshot_date)
        )
    ].copy()

    # If a customer has multiple eligible records, use the latest start.
    valid = valid.sort_values(
        ["customer_id", "subscription_start_date", "subscription_id"]
    )

    valid = valid.drop_duplicates("customer_id", keep="last")

    return valid[["customer_id", "plan_id", "monthly_price"]]


def run_simulation(
    contact_percentages,
    campaign_cost,
    retention_rate,
    months_saved,
):
    predictions = load_customer_predictions()

    snapshot_date = predictions["snapshot_date"].max()

    prices = load_snapshot_subscription_prices(snapshot_date)

    predictions = predictions.merge(
        prices,
        on="customer_id",
        how="left",
        validate="one_to_one",
    )

    total_customers = len(predictions)
    actual_churners = int(predictions["actual_churn"].sum())
    baseline_churn_rate = predictions["actual_churn"].mean()

    print("\n" + "=" * 65)
    print("NETFLIX CUSTOMER RETENTION — ROI SIMULATOR")
    print("=" * 65)
    print(f"Model: CatBoost_B")
    print(f"Snapshot date: {snapshot_date.date()}")
    print(f"Customers evaluated: {total_customers:,}")
    print(f"Actual churners: {actual_churners:,}")
    print(f"Baseline churn rate: {baseline_churn_rate:.2%}")
    print(f"Campaign cost per contact: {campaign_cost:.2f}")
    print(f"Assumed retention success rate: {retention_rate:.0%}")
    print(f"Revenue-saving period: {months_saved} months")

    print("\nSubscription price coverage:")
    print(f"Matched customers: {predictions['monthly_price'].notna().sum():,}")
    print(f"Missing prices: {predictions['monthly_price'].isna().sum():,}")

    results = []

    for percentage in contact_percentages:
        if not 0 < percentage <= 100:
            raise ValueError("Contact percentages must be between 0 and 100.")

        contact_count = max(
            1, round(total_customers * percentage / 100)
        )

        selected = predictions.head(contact_count).copy()

        actual_churners_selected = int(selected["actual_churn"].sum())

        # Retention success is an assumption, not a measured model outcome.
        expected_retained = (
            actual_churners_selected * retention_rate
        )

        # Estimate revenue only where a historical subscription price exists.
        revenue_eligible = selected[
            (selected["actual_churn"] == 1)
            & selected["monthly_price"].notna()
        ]

        expected_retained_revenue_customers = (
            revenue_eligible["monthly_price"].sum()
            * retention_rate
        )

        gross_revenue_saved = (
            expected_retained_revenue_customers * months_saved
        )

        campaign_total_cost = contact_count * campaign_cost
        net_benefit = gross_revenue_saved - campaign_total_cost

        roi_percent = (
            (net_benefit / campaign_total_cost) * 100
            if campaign_total_cost > 0
            else 0
        )

        precision = actual_churners_selected / contact_count
        recall = actual_churners_selected / actual_churners
        lift = (
            precision / baseline_churn_rate
            if baseline_churn_rate > 0
            else 0
        )

        results.append({
            "contact_capacity_pct": percentage,
            "customers_contacted": contact_count,
            "actual_churners_selected": actual_churners_selected,
            "expected_retained_customers": expected_retained,
            "precision": precision,
            "recall": recall,
            "lift": lift,
            "campaign_cost": campaign_total_cost,
            "estimated_gross_revenue_saved": gross_revenue_saved,
            "estimated_net_benefit": net_benefit,
            "estimated_roi_pct": roi_percent,
            "customers_with_price": len(revenue_eligible),
        })

    results_df = pd.DataFrame(results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(
        OUTPUT_DIR / "retention_roi_summary.csv", index=False
    )

    predictions.to_csv(
        OUTPUT_DIR / "catboost_b_ranked_customers_with_prices.csv",
        index=False,
    )

    print("\n" + "=" * 65)
    print("CAMPAIGN SCENARIO RESULTS")
    print("=" * 65)

    display_columns = [
        "contact_capacity_pct",
        "customers_contacted",
        "actual_churners_selected",
        "expected_retained_customers",
        "precision",
        "recall",
        "lift",
        "campaign_cost",
        "estimated_gross_revenue_saved",
        "estimated_net_benefit",
        "estimated_roi_pct",
    ]

    print(results_df[display_columns].to_string(
        index=False,
        formatters={
            "precision": "{:.2%}".format,
            "recall": "{:.2%}".format,
            "lift": "{:.2f}x".format,
            "expected_retained_customers": "{:.1f}".format,
            "campaign_cost": "{:.2f}".format,
            "estimated_gross_revenue_saved": "{:.2f}".format,
            "estimated_net_benefit": "{:.2f}".format,
            "estimated_roi_pct": "{:.2f}%".format,
        },
    ))

    print("\nReports saved to:", OUTPUT_DIR)


def main():
    parser = argparse.ArgumentParser(
        description="Simulate customer retention campaign ROI."
    )

    parser.add_argument(
        "--cost",
        type=float,
        default=1.00,
        help="Campaign cost per contacted customer.",
    )
    parser.add_argument(
        "--retention-rate",
        type=float,
        default=0.20,
        help="Assumed retention success rate, from 0 to 1.",
    )
    parser.add_argument(
        "--months-saved",
        type=int,
        default=3,
        help="Number of revenue months assumed to be saved.",
    )
    parser.add_argument(
        "--capacities",
        type=float,
        nargs="+",
        default=[5, 10, 15, 20],
        help="Contact capacity percentages.",
    )

    args = parser.parse_args()

    if args.cost < 0:
        parser.error("--cost cannot be negative.")

    if not 0 <= args.retention_rate <= 1:
        parser.error("--retention-rate must be between 0 and 1.")

    if args.months_saved < 1:
        parser.error("--months-saved must be at least 1.")

    run_simulation(
        contact_percentages=args.capacities,
        campaign_cost=args.cost,
        retention_rate=args.retention_rate,
        months_saved=args.months_saved,
    )


if __name__ == "__main__":
    main()
