
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
    "viewing_date_range": """
        SELECT
            MIN(viewing_date) AS earliest_viewing,
            MAX(viewing_date) AS latest_viewing,
            COUNT(*) AS total_records,
            COUNT(DISTINCT customer_id) AS unique_customers,
            COUNT(DISTINCT content_id) AS unique_content
        FROM viewing_activity
    """,
    "viewing_by_year": """
        SELECT
            EXTRACT(YEAR FROM viewing_date)::int AS year,
            COUNT(*) AS viewing_records,
            COUNT(DISTINCT customer_id) AS unique_customers,
            ROUND(AVG(watch_duration_minutes), 2) AS avg_watch_minutes
        FROM viewing_activity
        GROUP BY year
        ORDER BY year
    """,
    "viewing_by_churn_label": """
        SELECT
            cl.churned,
            COUNT(*) AS viewing_records,
            COUNT(DISTINCT v.customer_id) AS unique_customers,
            ROUND(AVG(v.watch_duration_minutes), 2)
                AS avg_watch_minutes,
            ROUND(AVG(v.completion_percentage), 2)
                AS avg_completion_percentage,
            ROUND(AVG(v.session_duration_minutes), 2)
                AS avg_session_minutes
        FROM viewing_activity v
        JOIN churn_labels cl
            ON cl.customer_id = v.customer_id
        GROUP BY cl.churned
        ORDER BY cl.churned
    """,
    "customers_without_viewing": """
        SELECT
            COUNT(*) AS customers_without_viewing
        FROM customers c
        LEFT JOIN viewing_activity v
            ON v.customer_id = c.customer_id
        WHERE v.customer_id IS NULL
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
