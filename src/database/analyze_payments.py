
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
    "payment_status_distribution": """
        SELECT payment_status, COUNT(*) AS records
        FROM payments
        GROUP BY payment_status
        ORDER BY records DESC
    """,
    "payment_status_by_churn": """
        SELECT
            cl.churned,
            p.payment_status,
            COUNT(*) AS payments,
            COUNT(DISTINCT p.customer_id) AS unique_customers,
            ROUND(AVG(p.amount), 2) AS average_amount
        FROM payments p
        JOIN churn_labels cl
            ON cl.customer_id = p.customer_id
        GROUP BY cl.churned, p.payment_status
        ORDER BY cl.churned, payments DESC
    """,
    "payment_date_range": """
        SELECT
            MIN(payment_date) AS earliest_payment,
            MAX(payment_date) AS latest_payment,
            COUNT(*) AS total_payments,
            COUNT(payment_date) AS payments_with_dates,
            COUNT(amount) AS payments_with_amounts
        FROM payments
    """,
    "payment_failure_reason": """
        SELECT
            failed_payment_reason,
            COUNT(*) AS records
        FROM payments
        WHERE failed_payment_reason IS NOT NULL
        GROUP BY failed_payment_reason
        ORDER BY records DESC
    """,
    "customers_without_payments": """
        SELECT COUNT(*) AS customers_without_payments
        FROM customers c
        LEFT JOIN payments p
            ON p.customer_id = c.customer_id
        WHERE p.customer_id IS NULL
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
