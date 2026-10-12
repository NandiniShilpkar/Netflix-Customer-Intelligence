
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


def main():
    queries = {
        "churn_distribution": """
            SELECT
                churned,
                COUNT(*) AS customers,
                ROUND(
                    100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
                    2
                ) AS percentage
            FROM churn_labels
            GROUP BY churned
            ORDER BY churned
        """,
        "label_duplicates": """
            SELECT customer_id, COUNT(*) AS records
            FROM churn_labels
            GROUP BY customer_id
            HAVING COUNT(*) > 1
            ORDER BY records DESC
            LIMIT 10
        """,
        "label_null_customer_ids": """
            SELECT COUNT(*) AS null_customer_ids
            FROM churn_labels
            WHERE customer_id IS NULL
        """,
        "customer_label_coverage": """
            SELECT
                (SELECT COUNT(*) FROM customers) AS total_customers,
                (SELECT COUNT(DISTINCT customer_id)
                 FROM churn_labels) AS customers_with_labels,
                (SELECT COUNT(*)
                 FROM customers c
                 LEFT JOIN churn_labels cl
                   ON c.customer_id = cl.customer_id
                 WHERE cl.customer_id IS NULL) AS customers_without_labels
        """,
        "label_date_consistency": """
            SELECT
                churned,
                COUNT(*) AS records,
                COUNT(churn_date) AS records_with_churn_date,
                COUNT(churn_reason) AS records_with_churn_reason
            FROM churn_labels
            GROUP BY churned
            ORDER BY churned
        """,
    }

    try:
        with engine.connect() as connection:
            for title, query in queries.items():
                print("\n" + "=" * 55)
                print(title.replace("_", " ").upper())
                print("=" * 55)

                result = pd.read_sql(text(query), connection)

                if result.empty:
                    print("No records found.")
                else:
                    print(result.to_string(index=False))

    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
