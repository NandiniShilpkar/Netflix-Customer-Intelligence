
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
    "issue_categories": """
        SELECT
            issue_category,
            COUNT(*) AS tickets,
            COUNT(DISTINCT customer_id) AS unique_customers
        FROM support_tickets
        GROUP BY issue_category
        ORDER BY tickets DESC
    """,
    "issue_subcategories": """
        SELECT
            issue_subcategory,
            COUNT(*) AS tickets
        FROM support_tickets
        GROUP BY issue_subcategory
        ORDER BY tickets DESC
        LIMIT 15
    """,
    "ticket_priorities": """
        SELECT priority, COUNT(*) AS tickets
        FROM support_tickets
        GROUP BY priority
        ORDER BY tickets DESC
    """,
    "ticket_statuses": """
        SELECT ticket_status, COUNT(*) AS tickets
        FROM support_tickets
        GROUP BY ticket_status
        ORDER BY tickets DESC
    """,
    "categories_by_churn": """
        SELECT
            cl.churned,
            st.issue_category,
            COUNT(*) AS tickets,
            COUNT(DISTINCT st.customer_id) AS unique_customers
        FROM support_tickets st
        JOIN churn_labels cl
            ON cl.customer_id = st.customer_id
        GROUP BY cl.churned, st.issue_category
        ORDER BY st.issue_category, cl.churned
    """,
    "resolution_and_satisfaction": """
        SELECT
            cl.churned,
            COUNT(*) AS tickets,
            ROUND(AVG(st.resolution_time_hours), 2)
                AS avg_resolution_hours,
            ROUND(AVG(st.customer_satisfaction_score), 2)
                AS avg_satisfaction,
            COUNT(st.resolution_time_hours) AS with_resolution_time,
            COUNT(st.customer_satisfaction_score)
                AS with_satisfaction_score
        FROM support_tickets st
        JOIN churn_labels cl
            ON cl.customer_id = st.customer_id
        GROUP BY cl.churned
        ORDER BY cl.churned
    """,
    "ticket_date_range": """
        SELECT
            MIN(ticket_date) AS earliest_ticket,
            MAX(ticket_date) AS latest_ticket,
            COUNT(*) AS total_tickets,
            COUNT(ticket_date) AS tickets_with_dates
        FROM support_tickets
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
