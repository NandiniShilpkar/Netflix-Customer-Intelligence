
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

database_url = os.getenv("DATABASE_URL")
if not database_url:
    raise RuntimeError("DATABASE_URL is missing from .env")

tables = [
    "customers",
    "subscription_plans",
    "subscriptions",
    "content",
    "viewing_activity",
    "payments",
    "support_tickets",
    "customer_feedback",
    "churn_labels",
]

engine = create_engine(
    database_url,
    pool_pre_ping=True,
    connect_args={"connect_timeout": 15},
)

try:
    with engine.connect() as connection:
        print("DATABASE ROW COUNTS")
        total = 0

        for table in tables:
            count = connection.execute(
                text(f'SELECT COUNT(*) FROM "{table}"')
            ).scalar_one()

            total += count
            print(f"{table:22} {count:>10,}")

        print("-" * 34)
        print(f"{'TOTAL':22} {total:>10,}")

finally:
    engine.dispose()
