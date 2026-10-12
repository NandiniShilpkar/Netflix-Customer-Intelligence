
import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is missing from .env")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

TABLES = [
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

# Likely relationship columns to inspect if present in each table.
RELATIONSHIPS = {
    "subscriptions": ["customer_id", "plan_id"],
    "viewing_activity": ["customer_id", "content_id"],
    "payments": ["customer_id", "subscription_id"],
    "support_tickets": ["customer_id"],
    "customer_feedback": ["customer_id", "content_id"],
    "churn_labels": ["customer_id"],
}


def main():
    inspector = inspect(engine)

    try:
        print("=" * 65)
        print("NETFLIX DATA QUALITY REPORT")
        print("=" * 65)

        # 1. Missing values and duplicate rows.
        for table in TABLES:
            df = pd.read_sql(text(f'SELECT * FROM "{table}"'), engine)

            print(f"\n--- {table} ---")
            print(f"Rows: {len(df):,}")
            print(f"Duplicate full rows: {df.duplicated().sum():,}")

            missing = df.isna().sum()
            missing = missing[missing > 0].sort_values(ascending=False)

            if missing.empty:
                print("Missing values: none")
            else:
                print("Columns with missing values:")
                for column, count in missing.items():
                    percentage = count / len(df) * 100 if len(df) else 0
                    print(
                        f"  {column}: {count:,} ({percentage:.2f}%)"
                    )

        # 2. Inspect actual database foreign keys.
        print("\n" + "=" * 65)
        print("DECLARED FOREIGN KEYS")
        print("=" * 65)

        for table in TABLES:
            foreign_keys = inspector.get_foreign_keys(table)
            print(f"\n{table}:")
            if not foreign_keys:
                print("  No declared foreign keys")
            for fk in foreign_keys:
                print(
                    f"  {fk['constrained_columns']} -> "
                    f"{fk['referred_table']}.{fk['referred_columns']}"
                )

        # 3. Check likely relationships using the actual column names.
        print("\n" + "=" * 65)
        print("RELATIONSHIP CHECKS")
        print("=" * 65)

        columns_by_table = {
            table: {column["name"] for column in inspector.get_columns(table)}
            for table in TABLES
        }

        checks = [
            ("subscriptions", "customer_id", "customers", "customer_id"),
            ("subscriptions", "plan_id", "subscription_plans", "plan_id"),
            ("viewing_activity", "customer_id", "customers", "customer_id"),
            ("viewing_activity", "content_id", "content", "content_id"),
            ("payments", "customer_id", "customers", "customer_id"),
            ("payments", "subscription_id", "subscriptions", "subscription_id"),
            ("support_tickets", "customer_id", "customers", "customer_id"),
            ("customer_feedback", "customer_id", "customers", "customer_id"),
            ("customer_feedback", "content_id", "content", "content_id"),
            ("churn_labels", "customer_id", "customers", "customer_id"),
        ]

        for source_table, source_col, target_table, target_col in checks:
            if (
                source_col not in columns_by_table[source_table]
                or target_col not in columns_by_table[target_table]
            ):
                print(
                    f"SKIPPED: {source_table}.{source_col} -> "
                    f"{target_table}.{target_col} "
                    "(column not present)"
                )
                continue

            query = text(
                f'''
                SELECT COUNT(*)
                FROM "{source_table}" AS s
                LEFT JOIN "{target_table}" AS t
                  ON s."{source_col}" = t."{target_col}"
                WHERE s."{source_col}" IS NOT NULL
                  AND t."{target_col}" IS NULL
                '''
            )

            with engine.connect() as connection:
                unmatched = connection.execute(query).scalar_one()

            print(
                f"{source_table}.{source_col} -> "
                f"{target_table}.{target_col}: "
                f"{unmatched:,} unmatched non-null values"
            )

        print("\nValidation report completed.")

    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
