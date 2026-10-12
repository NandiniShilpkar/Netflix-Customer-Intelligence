
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
    "subscription_status": """
        SELECT subscription_status, COUNT(*) AS records
        FROM subscriptions
        GROUP BY subscription_status
        ORDER BY records DESC
    """,
    "plan_distribution": """
        SELECT sp.plan_name, COUNT(*) AS subscriptions
        FROM subscriptions s
        JOIN subscription_plans sp ON s.plan_id = sp.plan_id
        GROUP BY sp.plan_name
        ORDER BY subscriptions DESC
    """,
    "customers_with_multiple_subscriptions": """
        SELECT
            COUNT(*) AS customers_with_subscriptions,
            COUNT(*) FILTER (WHERE subscription_count > 1)
                AS customers_with_multiple,
            MAX(subscription_count) AS maximum_per_customer
        FROM (
            SELECT customer_id, COUNT(*) AS subscription_count
            FROM subscriptions
            GROUP BY customer_id
        ) AS counts
    """,
    "churn_label_vs_subscription_status": """
        SELECT
            cl.churned,
            s.subscription_status,
            COUNT(DISTINCT cl.customer_id) AS customers
        FROM churn_labels cl
        JOIN subscriptions s ON s.customer_id = cl.customer_id
        GROUP BY cl.churned, s.subscription_status
        ORDER BY cl.churned, customers DESC
    """,
    "subscription_date_quality": """
        SELECT
            COUNT(*) AS total_subscriptions,
            COUNT(subscription_start_date) AS with_start_date,
            COUNT(subscription_end_date) AS with_end_date,
            COUNT(cancellation_date) AS with_cancellation_date,
            COUNT(*) FILTER (
                WHERE cancellation_date < subscription_start_date
            ) AS cancellation_before_start,
            COUNT(*) FILTER (
                WHERE subscription_end_date < subscription_start_date
            ) AS end_before_start
        FROM subscriptions
    """,
}

def main():
    try:
        with engine.connect() as connection:
            for name, query in QUERIES.items():
                print("\n" + "=" * 60)
                print(name.replace("_", " ").upper())
                print("=" * 60)

                result = pd.read_sql(text(query), connection)
                print(result.to_string(index=False))
    finally:
        engine.dispose()

if __name__ == "__main__":
    main()
