import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL missing. Check your .env file.")

engine = create_engine(DATABASE_URL)

checks = {
    "Q18_MISSING_REGISTRATION_MONTHS": """
        WITH months AS (
            SELECT DISTINCT DATE_TRUNC('month', registration_date)::date AS month
            FROM customers
            WHERE registration_date IS NOT NULL
        ),
        bounds AS (
            SELECT MIN(month) AS min_month, MAX(month) AS max_month
            FROM months
        ),
        calendar AS (
            SELECT GENERATE_SERIES(
                min_month, max_month, INTERVAL '1 month'
            )::date AS month
            FROM bounds
        )
        SELECT COUNT(*) AS missing_calendar_months
        FROM calendar c
        LEFT JOIN months m ON c.month = m.month
        WHERE m.month IS NULL;
    """,

    "Q19_NON_CONSECUTIVE_VIEWING_MONTHS": """
        WITH customer_months AS (
            SELECT DISTINCT
                customer_id,
                DATE_TRUNC('month', viewing_date)::date AS viewing_month
            FROM viewing_activity
            WHERE viewing_date IS NOT NULL
        ),
        ranked AS (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY customer_id
                    ORDER BY viewing_month DESC
                ) AS rn
            FROM customer_months
        ),
        latest AS (
            SELECT
                customer_id,
                MAX(viewing_month) FILTER (WHERE rn = 1) AS latest_month,
                MAX(viewing_month) FILTER (WHERE rn = 2) AS previous_month
            FROM ranked
            GROUP BY customer_id
        )
        SELECT
            COUNT(*) FILTER (
                WHERE previous_month IS NOT NULL
            ) AS customers_with_two_observed_months,
            COUNT(*) FILTER (
                WHERE previous_month IS NOT NULL
                  AND previous_month <>
                      (latest_month - INTERVAL '1 month')::date
            ) AS customers_with_nonconsecutive_months
        FROM latest;
    """,

    "Q20_COHORT_MATURITY": """
        WITH bounds AS (
            SELECT
                MAX(registration_date)::date AS latest_registration_date,
                (SELECT MAX(subscription_start_date)::date
                 FROM subscriptions) AS latest_subscription_date
            FROM customers
        )
        SELECT
            latest_registration_date,
            latest_subscription_date,
            COUNT(*) FILTER (
                WHERE registration_date::date + INTERVAL '1 month'
                      > latest_subscription_date
            ) AS customers_without_1_month_observation,
            COUNT(*) FILTER (
                WHERE registration_date::date + INTERVAL '3 months'
                      > latest_subscription_date
            ) AS customers_without_3_month_observation,
            COUNT(*) FILTER (
                WHERE registration_date::date + INTERVAL '6 months'
                      > latest_subscription_date
            ) AS customers_without_6_month_observation
        FROM customers, bounds
        WHERE registration_date IS NOT NULL
        GROUP BY latest_registration_date, latest_subscription_date;
    """,

    "Q30_NULL_CHURN_DATES": """
        SELECT
            COUNT(*) AS total_churned_customers,
            COUNT(*) FILTER (WHERE churn_date IS NULL)
                AS churned_customers_without_churn_date,
            COUNT(*) FILTER (WHERE churn_date IS NOT NULL)
                AS churned_customers_with_churn_date
        FROM churn_labels
        WHERE churned IS TRUE;
    """,

    "Q30_PAYMENTS_AFTER_CHURN": """
        SELECT COUNT(*) AS successful_payments_after_churn
        FROM churn_labels cl
        JOIN payments p ON p.customer_id = cl.customer_id
        WHERE cl.churned IS TRUE
          AND cl.churn_date IS NOT NULL
          AND p.payment_status = 'Successful'
          AND p.payment_date > cl.churn_date;
    """,

    "Q34_MIGRATION_COUNTS": """
        WITH migrations AS (
            SELECT
                customer_id,
                subscription_start_date,
                monthly_price,
                LAG(monthly_price) OVER (
                    PARTITION BY customer_id
                    ORDER BY subscription_start_date, subscription_id
                ) AS previous_price
            FROM subscriptions
        )
        SELECT
            COUNT(*) FILTER (WHERE previous_price IS NOT NULL)
                AS total_migration_events,
            COUNT(DISTINCT customer_id) FILTER (
                WHERE previous_price IS NOT NULL
            ) AS unique_customers_with_migrations,
            COUNT(*) FILTER (
                WHERE previous_price IS NOT NULL
                  AND monthly_price > previous_price
            ) AS upgrades,
            COUNT(*) FILTER (
                WHERE previous_price IS NOT NULL
                  AND monthly_price < previous_price
            ) AS downgrades,
            COUNT(*) FILTER (
                WHERE previous_price IS NOT NULL
                  AND monthly_price = previous_price
            ) AS same_price_migrations
        FROM migrations;
    """,

    "Q35_VIEWING_DATE_RANGE": """
        SELECT
            MIN(viewing_date)::date AS first_viewing_date,
            MAX(viewing_date)::date AS last_viewing_date,
            COUNT(DISTINCT DATE_TRUNC('week', viewing_date))
                AS distinct_calendar_weeks,
            COUNT(DISTINCT DATE_TRUNC('month', viewing_date))
                AS distinct_calendar_months
        FROM viewing_activity
        WHERE viewing_date IS NOT NULL;
    """
}

results = []

print("=" * 65)
print("DEEP BUSINESS LOGIC VALIDATION")
print("=" * 65)

with engine.connect() as conn:
    for name, query in checks.items():
        try:
            df = pd.read_sql(text(query), conn)
            print(f"\n{name}")
            print(df.to_string(index=False))

            for _, row in df.iterrows():
                for col, value in row.items():
                    results.append({
                        "check": name,
                        "metric": col,
                        "value": value
                    })

        except Exception as e:
            print(f"\n{name}: ERROR - {e}")
            results.append({
                "check": name,
                "metric": "ERROR",
                "value": str(e)
            })

output = ROOT / "data/reports/sql_question_results"
output.mkdir(parents=True, exist_ok=True)

report = pd.DataFrame(results)
report.to_csv(output / "deep_business_validation.csv", index=False)

print("\n" + "=" * 65)
print("VALIDATION COMPLETED")
print(f"Report: {output / 'deep_business_validation.csv'}")
print("=" * 65)
