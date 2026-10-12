
import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is missing from .env")

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    connect_args={"connect_timeout": 15},
)

QUERIES = {
    "invalid_cancellation_dates": """
        SELECT
            s.subscription_id,
            s.customer_id,
            s.subscription_status,
            s.subscription_start_date,
            s.cancellation_date,
            s.subscription_end_date,
            cl.churned,
            cl.churn_date,
            cl.churn_reason
        FROM subscriptions s
        LEFT JOIN churn_labels cl
            ON cl.customer_id = s.customer_id
        WHERE s.cancellation_date < s.subscription_start_date
        ORDER BY s.cancellation_date
        LIMIT 20
    """,
    "plan_changed_examples": """
        SELECT
            s.subscription_id,
            s.customer_id,
            s.subscription_status,
            s.subscription_start_date,
            s.subscription_end_date,
            s.cancellation_date,
            s.monthly_price,
            cl.churned,
            cl.churn_date
        FROM subscriptions s
        LEFT JOIN churn_labels cl
            ON cl.customer_id = s.customer_id
        WHERE s.subscription_status = 'Plan Changed'
        ORDER BY cl.churned, s.subscription_start_date
        LIMIT 20
    """,
    "status_counts_by_label": """
        SELECT
            cl.churned,
            s.subscription_status,
            COUNT(*) AS subscription_records,
            COUNT(DISTINCT s.customer_id) AS unique_customers
        FROM subscriptions s
        JOIN churn_labels cl
            ON cl.customer_id = s.customer_id
        GROUP BY cl.churned, s.subscription_status
        ORDER BY cl.churned, s.subscription_status
    """,
    "customer_subscription_counts": """
        SELECT
            subscription_count,
            COUNT(*) AS customers
        FROM (
            SELECT customer_id, COUNT(*) AS subscription_count
            FROM subscriptions
            GROUP BY customer_id
        ) customer_counts
        GROUP BY subscription_count
        ORDER BY subscription_count
    """,
}


def main():
    try:
        with engine.connect() as connection:
            for name, query in QUERIES.items():
                print("\n" + "=" * 65)
                print(name.replace("_", " ").upper())
                print("=" * 65)

                df = pd.read_sql(text(query), connection)
                if df.empty:
                    print("No records found.")
                else:
                    print(df.to_string(index=False))

    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
