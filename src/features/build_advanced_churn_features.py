import os
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

INPUT_FILE = os.path.join(
    BASE_DIR, "data/features/churn_historical_snapshots.csv"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "data/features")
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR, "churn_advanced_snapshots.csv"
)

load_dotenv(os.path.join(BASE_DIR, ".env"))
engine = create_engine(os.environ["DATABASE_URL"])


def load_activity_features(conn, cutoff):
    cutoff = pd.Timestamp(cutoff)
    current_start = cutoff - pd.Timedelta(days=30)
    previous_start = cutoff - pd.Timedelta(days=60)

    params = {
        "cutoff": cutoff.date(),
        "current_start": current_start.date(),
        "previous_start": previous_start.date()
    }

    # 1. Viewing behaviour: current 30 days vs previous 30 days
    viewing_query = text("""
        SELECT
            customer_id,

            COUNT(*) FILTER (
                WHERE viewing_date > :current_start
            ) AS viewing_sessions_30d,

            COUNT(DISTINCT viewing_date) FILTER (
                WHERE viewing_date > :current_start
            ) AS active_viewing_days_30d,

            COALESCE(SUM(watch_duration_minutes) FILTER (
                WHERE viewing_date > :current_start
            ), 0) AS watch_minutes_30d,

            AVG(completion_percentage) FILTER (
                WHERE viewing_date > :current_start
            ) AS avg_completion_30d,

            AVG(session_duration_minutes) FILTER (
                WHERE viewing_date > :current_start
            ) AS avg_session_duration_30d,

            COUNT(*) FILTER (
                WHERE viewing_date > :previous_start
                  AND viewing_date <= :current_start
            ) AS viewing_sessions_previous_30d,

            COUNT(DISTINCT viewing_date) FILTER (
                WHERE viewing_date > :previous_start
                  AND viewing_date <= :current_start
            ) AS active_viewing_days_previous_30d,

            COALESCE(SUM(watch_duration_minutes) FILTER (
                WHERE viewing_date > :previous_start
                  AND viewing_date <= :current_start
            ), 0) AS watch_minutes_previous_30d,

            COUNT(DISTINCT LOWER(TRIM(device_type))) FILTER (
                WHERE viewing_date > :current_start
                  AND device_type IS NOT NULL
            ) AS device_types_used_30d

        FROM viewing_activity
        WHERE viewing_date > :previous_start
          AND viewing_date <= :cutoff
        GROUP BY customer_id
    """)

    viewing = pd.read_sql(viewing_query, conn, params=params)

    # 2. Payment behaviour: current vs previous 30 days
    payment_query = text("""
        SELECT
            customer_id,

            COUNT(*) FILTER (
                WHERE payment_date > :current_start
            ) AS payments_30d,

            COUNT(*) FILTER (
                WHERE payment_date > :current_start
                  AND LOWER(TRIM(payment_status)) = 'failed'
            ) AS failed_payments_30d,

            COUNT(*) FILTER (
                WHERE payment_date > :current_start
                  AND LOWER(TRIM(payment_status)) = 'success'
            ) AS successful_payments_30d,

            COUNT(*) FILTER (
                WHERE payment_date > :previous_start
                  AND payment_date <= :current_start
                  AND LOWER(TRIM(payment_status)) = 'failed'
            ) AS failed_payments_previous_30d,

            COUNT(*) FILTER (
                WHERE payment_date > :previous_start
                  AND payment_date <= :current_start
            ) AS payments_previous_30d,

            COALESCE(SUM(amount) FILTER (
                WHERE payment_date > :current_start
                  AND LOWER(TRIM(payment_status)) = 'success'
            ), 0) AS successful_payment_amount_30d

        FROM payments
        WHERE payment_date > :previous_start
          AND payment_date <= :cutoff
        GROUP BY customer_id
    """)

    payments = pd.read_sql(payment_query, conn, params=params)

    # 3. Support behaviour
    support_query = text("""
        SELECT
            customer_id,

            COUNT(*) FILTER (
                WHERE ticket_date > :current_start
            ) AS support_tickets_30d,

            COUNT(*) FILTER (
                WHERE ticket_date > :current_start
                  AND LOWER(TRIM(ticket_status))
                      IN ('pending', 'escalated')
            ) AS unresolved_tickets_30d,

            COUNT(*) FILTER (
                WHERE ticket_date > :current_start
                  AND LOWER(TRIM(priority))
                      IN ('high', 'critical')
            ) AS high_priority_tickets_30d,

            COUNT(*) FILTER (
                WHERE ticket_date > :current_start
                  AND LOWER(TRIM(ticket_status)) = 'escalated'
            ) AS escalated_tickets_30d,

            AVG(customer_satisfaction_score) FILTER (
                WHERE ticket_date > :current_start
            ) AS avg_satisfaction_30d,

            AVG(resolution_time_hours) FILTER (
                WHERE ticket_date > :current_start
            ) AS avg_resolution_hours_30d,

            COUNT(*) FILTER (
                WHERE ticket_date > :previous_start
                  AND ticket_date <= :current_start
            ) AS support_tickets_previous_30d

        FROM support_tickets
        WHERE ticket_date > :previous_start
          AND ticket_date <= :cutoff
        GROUP BY customer_id
    """)

    support = pd.read_sql(support_query, conn, params=params)

    return viewing, payments, support


def safe_percentage_change(current, previous):
    current = pd.to_numeric(current, errors="coerce")
    previous = pd.to_numeric(previous, errors="coerce")

    result = np.where(
        previous > 0,
        ((current - previous) / previous) * 100,
        np.where(current > 0, 100.0, 0.0)
    )

    return result


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    base = pd.read_csv(INPUT_FILE)
    base["snapshot_date"] = pd.to_datetime(base["snapshot_date"])

    all_snapshots = []

    try:
        with engine.connect() as conn:
            for cutoff in sorted(base["snapshot_date"].unique()):
                cutoff = pd.Timestamp(cutoff)

                print(f"\nProcessing snapshot: {cutoff.date()}")

                snapshot = base[
                    base["snapshot_date"] == cutoff
                ].copy()

                viewing, payments, support = load_activity_features(
                    conn, cutoff
                )

                for extra in [viewing, payments, support]:
                    snapshot = snapshot.merge(
                        extra,
                        on="customer_id",
                        how="left",
                        validate="one_to_one"
                    )

                # Missing activity means zero recorded activity.
                count_columns = [
                    "viewing_sessions_30d",
                    "active_viewing_days_30d",
                    "watch_minutes_30d",
                    "viewing_sessions_previous_30d",
                    "active_viewing_days_previous_30d",
                    "watch_minutes_previous_30d",
                    "device_types_used_30d",
                    "payments_30d",
                    "failed_payments_30d",
                    "successful_payments_30d",
                    "failed_payments_previous_30d",
                    "payments_previous_30d",
                    "successful_payment_amount_30d",
                    "support_tickets_30d",
                    "unresolved_tickets_30d",
                    "high_priority_tickets_30d",
                    "escalated_tickets_30d",
                    "support_tickets_previous_30d"
                ]

                for col in count_columns:
                    snapshot[col] = snapshot[col].fillna(0)

                # Recent payment failure rate.
                snapshot["failed_payment_rate_30d"] = np.where(
                    snapshot["payments_30d"] > 0,
                    snapshot["failed_payments_30d"]
                    / snapshot["payments_30d"],
                    0
                )

                # Recent support risk indicator.
                snapshot["has_unresolved_support_30d"] = (
                    snapshot["unresolved_tickets_30d"] > 0
                ).astype(int)

                snapshot["has_critical_support_30d"] = (
                    snapshot["high_priority_tickets_30d"] > 0
                ).astype(int)

                # Engagement trends.
                snapshot["watch_minutes_change_pct"] = (
                    safe_percentage_change(
                        snapshot["watch_minutes_30d"],
                        snapshot["watch_minutes_previous_30d"]
                    )
                )

                snapshot["viewing_sessions_change_pct"] = (
                    safe_percentage_change(
                        snapshot["viewing_sessions_30d"],
                        snapshot["viewing_sessions_previous_30d"]
                    )
                )

                snapshot["active_days_change_pct"] = (
                    safe_percentage_change(
                        snapshot["active_viewing_days_30d"],
                        snapshot["active_viewing_days_previous_30d"]
                    )
                )

                snapshot["payment_failures_change"] = (
                    snapshot["failed_payments_30d"]
                    - snapshot["failed_payments_previous_30d"]
                )

                snapshot["support_ticket_change"] = (
                    snapshot["support_tickets_30d"]
                    - snapshot["support_tickets_previous_30d"]
                )

                # Behavioural warning flags.
                snapshot["declining_engagement_flag"] = (
                    (snapshot["watch_minutes_previous_30d"] > 0)
                    & (snapshot["watch_minutes_change_pct"] <= -30)
                ).astype(int)

                snapshot["payment_risk_flag"] = (
                    (snapshot["failed_payments_30d"] >= 2)
                    | (snapshot["failed_payment_rate_30d"] >= 0.5)
                ).astype(int)

                snapshot["support_risk_flag"] = (
                    (snapshot["unresolved_tickets_30d"] > 0)
                    | (snapshot["escalated_tickets_30d"] > 0)
                ).astype(int)

                all_snapshots.append(snapshot)

                print(
                    f"Customers: {len(snapshot):,} | "
                    f"Declining engagement: "
                    f"{snapshot['declining_engagement_flag'].sum():,} | "
                    f"Payment risk: "
                    f"{snapshot['payment_risk_flag'].sum():,} | "
                    f"Support risk: "
                    f"{snapshot['support_risk_flag'].sum():,}"
                )

    finally:
        engine.dispose()

    result = pd.concat(all_snapshots, ignore_index=True)

    result.to_csv(OUTPUT_FILE, index=False)

    print("\nADVANCED FEATURE ENGINEERING COMPLETED")
    print("Output:", OUTPUT_FILE)
    print("Dataset shape:", result.shape)
    print("Unique customers:", result["customer_id"].nunique())
    print(
        "Duplicate customer-snapshot pairs:",
        result.duplicated(["customer_id", "snapshot_date"]).sum()
    )
    print(
        "Target distribution:\n",
        result["churn_next_30_days"].value_counts().to_string()
    )


if __name__ == "__main__":
    main()
