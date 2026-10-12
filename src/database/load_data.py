
import os
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import MetaData, create_engine, insert, text

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is missing from .env")

DATA_DIR = Path("data/processed")

LOAD_ORDER = [
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

DATE_COLUMNS = {
    "customers": ["registration_date"],
    "subscriptions": [
        "subscription_start_date",
        "subscription_end_date",
        "cancellation_date",
    ],
    "viewing_activity": ["viewing_date"],
    "payments": ["payment_date"],
    "support_tickets": ["ticket_date"],
    "customer_feedback": ["feedback_date"],
    "churn_labels": ["churn_date"],
}

INTEGER_COLUMNS = {
    "customers": ["age"],
    "subscription_plans": ["max_devices"],
    "content": ["release_year", "duration_minutes"],
    "viewing_activity": [
        "watch_duration_minutes",
        "session_duration_minutes",
    ],
    "support_tickets": ["customer_satisfaction_score"],
    "customer_feedback": ["rating"],
}

NUMERIC_COLUMNS = {
    "subscription_plans": ["monthly_price"],
    "subscriptions": ["monthly_price"],
    "payments": ["amount"],
    "viewing_activity": ["completion_percentage"],
    "support_tickets": ["resolution_time_hours"],
}

BOOLEAN_COLUMNS = {
    "subscriptions": ["auto_renew"],
    "churn_labels": ["churned"],
}


def clean_value(value):
    if value is None or pd.isna(value):
        return None

    if isinstance(value, pd.Timestamp):
        return value.date()

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    if hasattr(value, "item"):
        return value.item()

    return value


def main():
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 15},
    )

    try:
        metadata = MetaData()
        metadata.reflect(bind=engine, only=LOAD_ORDER)
        tables = {name: metadata.tables[name] for name in LOAD_ORDER}

        # Refuse to append data to tables that already contain rows.
        with engine.connect() as connection:
            counts = {
                name: connection.execute(
                    text(f'SELECT COUNT(*) FROM "{name}"')
                ).scalar_one()
                for name in LOAD_ORDER
            }

        populated = {
            name: count for name, count in counts.items() if count > 0
        }
        if populated:
            raise RuntimeError(
                f"Tables are not empty; no data loaded: {populated}"
            )

        # Validate every CSV before writing anything to the database.
        frames = {}

        for name in LOAD_ORDER:
            path = DATA_DIR / f"{name}.csv"
            if not path.exists():
                raise FileNotFoundError(f"Missing CSV file: {path}")

            df = pd.read_csv(path)

            expected = set(tables[name].columns.keys())
            actual = set(df.columns)

            if actual != expected:
                raise ValueError(
                    f"{name} column mismatch. "
                    f"Missing: {sorted(expected - actual)}; "
                    f"Unexpected: {sorted(actual - expected)}"
                )

            for col in DATE_COLUMNS.get(name, []):
                parsed = pd.to_datetime(df[col], errors="coerce")
                df[col] = parsed.dt.date
                df[col] = df[col].map(
                    lambda value: None if pd.isna(value) else value
                )

            for col in INTEGER_COLUMNS.get(name, []):
                values = pd.to_numeric(df[col], errors="coerce")
                non_integer = values.notna() & (values % 1 != 0)

                if non_integer.any():
                    raise ValueError(
                        f"{name}.{col} contains non-integer values"
                    )

                # Convert valid numbers to Python int and missing values to None.
                df[col] = values.map(
                    lambda value: None
                    if pd.isna(value)
                    else int(value)
                )

            for col in NUMERIC_COLUMNS.get(name, []):
                values = pd.to_numeric(df[col], errors="coerce")
                df[col] = values.map(
                    lambda value: None
                    if pd.isna(value)
                    else float(value)
                )

            for col in BOOLEAN_COLUMNS.get(name, []):
                def parse_boolean(value):
                    if pd.isna(value):
                        return None
                    if isinstance(value, bool):
                        return value

                    normalized = str(value).strip().lower()
                    if normalized in {"true", "1", "yes"}:
                        return True
                    if normalized in {"false", "0", "no"}:
                        return False

                    raise ValueError(
                        f"Unexpected boolean in {name}.{col}: {value!r}"
                    )

                df[col] = df[col].map(parse_boolean)

            frames[name] = df

            print(
                f"Validated {name}: {len(df):,} rows, "
                f"{len(df.columns)} columns"
            )

        # One transaction ensures a failed load is rolled back.
        total = 0

        with engine.begin() as connection:
            for name in LOAD_ORDER:
                df = frames[name]
                records = [
                    {
                        column: clean_value(value)
                        for column, value in row.items()
                    }
                    for row in df.to_dict(orient="records")
                ]

                for start in range(0, len(records), 500):
                    batch = records[start:start + 500]
                    if batch:
                        connection.execute(insert(tables[name]), batch)

                total += len(records)
                print(f"Loaded {name}: {len(records):,} rows")

        print(f"\nSUCCESS: Loaded {total:,} total rows.")

    except Exception as error:
        print(f"\nDATA LOAD FAILED: {type(error).__name__}: {error}")
        print("The database transaction should roll back on failure.")
        raise

    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
