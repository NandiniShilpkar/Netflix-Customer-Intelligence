
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CUTOFF_DATE = "2026-03-31"
CHURN_WINDOW_END = "2026-04-30"
VIEWING_WINDOW_START = "2025-12-31"

load_dotenv(PROJECT_ROOT / ".env")

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is missing. Check your .env file.")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)


# --------------------------------------------------
# 2. Load customer cohort and target
# --------------------------------------------------


cohort_query = text("""
    WITH latest_subscription AS (
        SELECT DISTINCT ON (customer_id)
            customer_id,
            subscription_start_date,
            plan_id,
            cancellation_date
        FROM subscriptions
        WHERE subscription_start_date <= CAST(:cutoff AS DATE)
        ORDER BY customer_id, subscription_start_date DESC
    )
    SELECT
        c.customer_id,
        c.age,
        c.gender,
        c.country,
        c.city,
        c.preferred_language,
        ls.plan_id,
        ls.subscription_start_date,
        CASE
            WHEN cl.churned = TRUE
             AND cl.churn_date > CAST(:cutoff AS DATE)
             AND cl.churn_date <= CAST(:window_end AS DATE)
            THEN 1
            ELSE 0
        END AS churn_next_30_days
    FROM customers c
    JOIN latest_subscription ls
        ON c.customer_id = ls.customer_id
    JOIN churn_labels cl
        ON c.customer_id = cl.customer_id
    WHERE (
        ls.cancellation_date IS NULL
        OR ls.cancellation_date > CAST(:cutoff AS DATE)
    )
      AND (
          cl.churned = FALSE
          OR cl.churn_date > CAST(:cutoff AS DATE)
      )
""")


params = {
    "cutoff": CUTOFF_DATE,
    "window_end": CHURN_WINDOW_END,
}

with engine.connect() as conn:
    features = pd.read_sql(cohort_query, conn, params=params)

print(f"Eligible customers: {len(features):,}")


# --------------------------------------------------
# 3. Viewing activity features: preceding 90 days
# --------------------------------------------------

viewing_query = text("""
    SELECT
        customer_id,
        COUNT(*) AS viewing_sessions_90d,
        COUNT(DISTINCT viewing_date) AS active_viewing_days_90d,
        COALESCE(SUM(watch_duration_minutes), 0) AS total_watch_minutes_90d,
        COALESCE(AVG(watch_duration_minutes), 0) AS avg_watch_minutes_90d,
        COALESCE(AVG(completion_percentage), 0) AS avg_completion_pct_90d,
        MAX(viewing_date) AS last_viewing_date
    FROM viewing_activity
    WHERE viewing_date > CAST(:window_start AS DATE)
      AND viewing_date <= CAST(:cutoff AS DATE)
    GROUP BY customer_id
""")

with engine.connect() as conn:
    viewing = pd.read_sql(
        viewing_query,
        conn,
        params={"window_start": VIEWING_WINDOW_START, "cutoff": CUTOFF_DATE},
    )

features = features.merge(viewing, on="customer_id", how="left")

viewing_numeric = [
    "viewing_sessions_90d",
    "active_viewing_days_90d",
    "total_watch_minutes_90d",
    "avg_watch_minutes_90d",
    "avg_completion_pct_90d",
]

features[viewing_numeric] = features[viewing_numeric].fillna(0)

features["days_since_last_viewing"] = (
    pd.Timestamp(CUTOFF_DATE)
    - pd.to_datetime(features["last_viewing_date"])
).dt.days

features["days_since_last_viewing"] = (
    features["days_since_last_viewing"].fillna(91)
)


# --------------------------------------------------
# 4. Payment features: history through cutoff
# --------------------------------------------------

payments_query = text("""
    SELECT
        customer_id,
        COUNT(*) AS payment_count,
        COUNT(*) FILTER (
            WHERE payment_status = 'Failed'
        ) AS failed_payment_count,
        COUNT(*) FILTER (
            WHERE payment_status = 'Success'
        ) AS successful_payment_count,
        COALESCE(AVG(amount), 0) AS avg_payment_amount
    FROM payments
    WHERE payment_date <= CAST(:cutoff AS DATE)
    GROUP BY customer_id
""")

with engine.connect() as conn:
    payments = pd.read_sql(
        payments_query,
        conn,
        params={"cutoff": CUTOFF_DATE},
    )

features = features.merge(payments, on="customer_id", how="left")

payment_numeric = [
    "payment_count",
    "failed_payment_count",
    "successful_payment_count",
    "avg_payment_amount",
]

features[payment_numeric] = features[payment_numeric].fillna(0)

features["failed_payment_rate"] = (
    features["failed_payment_count"]
    / features["payment_count"].replace(0, pd.NA)
).fillna(0)


# --------------------------------------------------
# 5. Support ticket features: history through cutoff
# --------------------------------------------------

support_query = text("""
    SELECT
        customer_id,
        COUNT(*) AS support_ticket_count,
        COUNT(*) FILTER (
            WHERE priority IN ('High', 'Critical')
        ) AS high_priority_ticket_count,
        COUNT(*) FILTER (
            WHERE ticket_status IN ('Pending', 'Escalated')
        ) AS unresolved_ticket_count,
        COALESCE(AVG(customer_satisfaction_score), 0)
            AS avg_support_satisfaction,
        COALESCE(AVG(resolution_time_hours), 0)
            AS avg_resolution_hours
    FROM support_tickets
    WHERE ticket_date <= CAST(:cutoff AS DATE)
    GROUP BY customer_id
""")

with engine.connect() as conn:
    support = pd.read_sql(
        support_query,
        conn,
        params={"cutoff": CUTOFF_DATE},
    )

features = features.merge(support, on="customer_id", how="left")

support_numeric = [
    "support_ticket_count",
    "high_priority_ticket_count",
    "unresolved_ticket_count",
    "avg_support_satisfaction",
    "avg_resolution_hours",
]

# A missing score is not the same as a score of zero.
# Keep missing averages as NaN for later imputation.
support_count_features = [
    "support_ticket_count",
    "high_priority_ticket_count",
    "unresolved_ticket_count",
]

features[support_count_features] = (
    features[support_count_features].fillna(0)
)

features["has_support_tickets"] = (
    features["support_ticket_count"] > 0
).astype(int)


# --------------------------------------------------
# 6. Clean and save dataset
# --------------------------------------------------

# Do not use cancellation dates, churn dates, churn reasons,
# or subscription status as model features.

features["subscription_tenure_days"] = (
    pd.Timestamp(CUTOFF_DATE)
    - pd.to_datetime(features["subscription_start_date"])
).dt.days

features = features.drop(
    columns=["subscription_start_date", "last_viewing_date"],
    errors="ignore",
)

features = features.drop_duplicates(subset=["customer_id"])

output_path = OUTPUT_DIR / "churn_features.csv"
features.to_csv(output_path, index=False)

print("\nFeature dataset created.")
print(f"Output: {output_path}")
print(f"Rows: {len(features):,}")
print(f"Columns: {len(features.columns):,}")
print(f"April churn cases: {features['churn_next_30_days'].sum():,}")
print("\nTarget distribution:")
print(features["churn_next_30_days"].value_counts(dropna=False))
print("\nFirst five rows:")
print(features.head().to_string(index=False))

engine.dispose()
