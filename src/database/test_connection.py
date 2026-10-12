
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

database_url = os.getenv("DATABASE_URL")

if not database_url:
    raise RuntimeError(
        "DATABASE_URL is missing. Check your .env file."
    )

engine = create_engine(
    database_url,
    pool_pre_ping=True,
    connect_args={"connect_timeout": 15},
)

try:
    with engine.connect() as connection:
        result = connection.execute(
            text("SELECT current_database(), current_user, version()")
        ).fetchone()

        print("Connection successful!")
        print("Database:", result[0])
        print("User:", result[1])
        print("PostgreSQL version:", result[2].split(",")[0])

except Exception as error:
    print("Connection failed:", type(error).__name__)
    print("Check your connection string, credentials, and Neon status.")

finally:
    engine.dispose()
