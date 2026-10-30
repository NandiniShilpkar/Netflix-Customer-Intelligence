from pathlib import Path
import pandas as pd

INPUT_FILE = Path(
    "reports/retention_roi/catboost_b_ranked_customers_with_prices.csv"
)
OUTPUT_DIR = Path("reports/retention_roi")

CAPACITIES = [5, 10, 15, 20]
RETENTION_RATES = [0.10, 0.20, 0.30, 0.40, 0.50]
CAMPAIGN_COSTS = [0.50, 1.00, 1.50, 2.00]
MONTHS_SAVED = [1, 3, 6]


def main():
    df = pd.read_csv(INPUT_FILE)

    df = df.sort_values(
        "predicted_churn_probability",
        ascending=False
    ).reset_index(drop=True)

    df["monthly_price"] = pd.to_numeric(
        df["monthly_price"], errors="coerce"
    )

    total_customers = len(df)
    scenarios = []

    for capacity in CAPACITIES:
        contact_count = max(
            1, round(total_customers * capacity / 100)
        )

        selected = df.head(contact_count)

        churners = selected[
            (selected["actual_churn"] == 1)
            & selected["monthly_price"].notna()
        ]

        churners_selected = len(churners)
        monthly_revenue_at_risk = churners["monthly_price"].sum()

        for retention_rate in RETENTION_RATES:
            for campaign_cost in CAMPAIGN_COSTS:
                for months in MONTHS_SAVED:

                    expected_retained = (
                        churners_selected * retention_rate
                    )

                    gross_revenue_saved = (
                        monthly_revenue_at_risk
                        * retention_rate
                        * months
                    )

                    total_campaign_cost = (
                        contact_count * campaign_cost
                    )

                    net_benefit = (
                        gross_revenue_saved - total_campaign_cost
                    )

                    roi = (
                        net_benefit / total_campaign_cost * 100
                        if total_campaign_cost > 0
                        else 0
                    )

                    break_even_rate = (
                        total_campaign_cost / (
                            monthly_revenue_at_risk * months
                        )
                        if monthly_revenue_at_risk > 0
                        else None
                    )

                    scenarios.append({
                        "contact_capacity_pct": capacity,
                        "customers_contacted": contact_count,
                        "actual_churners_selected": churners_selected,
                        "retention_success_rate": retention_rate,
                        "campaign_cost_per_customer": campaign_cost,
                        "revenue_saving_months": months,
                        "expected_retained_customers": expected_retained,
                        "monthly_revenue_at_risk": monthly_revenue_at_risk,
                        "estimated_gross_revenue_saved": gross_revenue_saved,
                        "total_campaign_cost": total_campaign_cost,
                        "estimated_net_benefit": net_benefit,
                        "estimated_roi_pct": roi,
                        "break_even_retention_rate": break_even_rate,
                        "profitable": net_benefit > 0,
                    })

    results = pd.DataFrame(scenarios)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / "retention_sensitivity_results.csv"
    results.to_csv(output_file, index=False)

    profitable = results[results["profitable"]].copy()

    best_scenarios = results.sort_values(
        "estimated_roi_pct", ascending=False
    ).head(10)

    profitable_summary = (
        results.groupby(
            ["contact_capacity_pct", "retention_success_rate"],
            as_index=False
        )
        .agg(
            profitable_scenarios=("profitable", "sum"),
            total_scenarios=("profitable", "count"),
            average_roi_pct=("estimated_roi_pct", "mean"),
        )
    )

    profitable_summary.to_csv(
        OUTPUT_DIR / "retention_sensitivity_summary.csv",
        index=False
    )

    best_scenarios.to_csv(
        OUTPUT_DIR / "top_10_retention_scenarios.csv",
        index=False
    )

    print("\n" + "=" * 65)
    print("RETENTION ROI SENSITIVITY ANALYSIS")
    print("=" * 65)

    print(f"Customers evaluated: {total_customers:,}")
    print(f"Total scenarios: {len(results):,}")
    print(f"Profitable scenarios: {len(profitable):,}")
    print(
        f"Unprofitable scenarios: "
        f"{len(results) - len(profitable):,}"
    )

    print("\nTOP 10 SCENARIOS BY ESTIMATED ROI")
    print(
        best_scenarios[
            [
                "contact_capacity_pct",
                "retention_success_rate",
                "campaign_cost_per_customer",
                "revenue_saving_months",
                "estimated_net_benefit",
                "estimated_roi_pct",
            ]
        ].to_string(
            index=False,
            formatters={
                "retention_success_rate": "{:.0%}".format,
                "campaign_cost_per_customer": "{:.2f}".format,
                "estimated_net_benefit": "{:.2f}".format,
                "estimated_roi_pct": "{:.2f}%".format,
            }
        )
    )

    print("\nFiles saved:")
    print(output_file)
    print(OUTPUT_DIR / "retention_sensitivity_summary.csv")
    print(OUTPUT_DIR / "top_10_retention_scenarios.csv")

    print(
        "\nNOTE: Retention success rates and campaign costs are "
        "assumptions, not measured campaign outcomes."
    )


if __name__ == "__main__":
    main()
