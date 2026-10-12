
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

database_url = os.getenv("DATABASE_URL")
if not database_url:
    raise RuntimeError("DATABASE_URL is missing from .env")

engine = create_engine(
    database_url,
    pool_pre_ping=True,
    connect_args={"connect_timeout": 15},
)

try:
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

        print(f"Existing tables: {len(tables)}")
        for table in tables:
            print(f"- {table}")

        print("\nColumn details:")
        columns = connection.execute(
            text("""
                SELECT table_name, column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'public'
                ORDER BY table_name, ordinal_position
            """)
        ).all()

        for table, column, data_type in columns:
            print(f"{table}.{column}: {data_type}")

finally:
    engine.dispose()
