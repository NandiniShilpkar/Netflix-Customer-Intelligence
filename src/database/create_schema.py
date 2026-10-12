
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import os

load_dotenv()

database_url = os.getenv("DATABASE_URL")
if not database_url:
    raise RuntimeError("DATABASE_URL is missing from .env")

schema_path = Path("sql/schema.sql")
if not schema_path.exists():
    raise FileNotFoundError(f"Schema file not found: {schema_path}")

schema_sql = schema_path.read_text(encoding="utf-8")

engine = create_engine(
    database_url,
    pool_pre_ping=True,
    connect_args={"connect_timeout": 15},
)

try:
    with engine.begin() as connection:
        connection.exec_driver_sql(schema_sql)

    print("Schema created successfully!")

    with engine.connect() as connection:
        tables = connection.execute(
            text("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """)
        ).scalars().all()

    print(f"Tables found: {len(tables)}")
    for table in tables:
        print(f" - {table}")

except Exception as error:
    print("Schema setup failed:", type(error).__name__)
    print("Review the SQL statement and database error before retrying.")
    raise

finally:
    engine.dispose()
