import os
import re

import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import sqlglot

from sqlglot import exp
from sqlalchemy import text


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")

MAX_ROWS = 200

ALLOWED_TABLES = {
    "customers",
    "subscriptions",
    "subscription_plans",
    "viewing_activity",
    "content",
    "payments",
    "support_tickets",
    "customer_feedback",
    "churn_labels",
}

DATABASE_SCHEMA = """
PostgreSQL database: Netflix Customer Intelligence

Tables and columns:

customers:
customer_id, first_name, last_name, age, gender, country,
state_or_region, city, registration_date, acquisition_channel,
customer_segment, preferred_language

subscriptions:
subscription_id, customer_id, plan_id, subscription_start_date,
subscription_end_date, subscription_status, auto_renew,
cancellation_date, cancellation_reason, monthly_price

subscription_plans:
plan_id, plan_name, monthly_price, video_quality,
max_devices, advertisements

viewing_activity:
viewing_id, customer_id, content_id, viewing_date,
watch_duration_minutes, completion_percentage, device_type,
login_location, session_duration_minutes

content:
content_id, title, content_type, genre, release_year,
duration_minutes, content_language, production_country,
maturity_rating

payments:
payment_id, customer_id, subscription_id, payment_date,
amount, payment_method, payment_status, failed_payment_reason

support_tickets:
ticket_id, customer_id, ticket_date, issue_category,
issue_subcategory, priority, ticket_status,
resolution_time_hours, customer_satisfaction_score,
ticket_description, support_agent_id

customer_feedback:
feedback_id, customer_id, feedback_date, rating,
feedback_text, sentiment_label

churn_labels:
customer_id, churned, churn_date, churn_reason

Relationships:
customers.customer_id connects to customer_id in the other
customer tables.

subscriptions.plan_id connects to subscription_plans.plan_id.
viewing_activity.content_id connects to content.content_id.
payments.subscription_id connects to subscriptions.subscription_id.

Important business rules:
- Successful payment records use payment_status = 'Success'.
- churned is a Boolean field in churn_labels.
- subscription_end_date can be NULL for ongoing subscriptions.
- A customer can have multiple historical subscription records.
- Do not assume that every subscription record is a unique customer.
- Use DISTINCT when the question asks for unique customers.
- Use PostgreSQL syntax.
"""


def generate_sql(question):
    """Generate PostgreSQL SQL from a natural-language question."""

    if not question or not question.strip():
        raise ValueError("Please enter a business question.")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing.")

    instructions = """
You are a PostgreSQL analytics SQL generator.

Return exactly one read-only SELECT query.
Do not return Markdown, explanations, or multiple statements.

Use only the approved schema and columns supplied below.
Never invent table names or column names.
Never generate INSERT, UPDATE, DELETE, DROP, ALTER, CREATE,
TRUNCATE, GRANT, or other write operations.
Use correct joins and PostgreSQL syntax.
Use DISTINCT when counting unique customers.
Do not treat historical subscription records as unique customers.
Do not call arbitrary PostgreSQL functions.
Use a LIMIT of 200 or less.

DATABASE SCHEMA:
""" + DATABASE_SCHEMA

    client = OpenAI()

    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions=instructions,
        input="Business question:\n" + question.strip(),
        max_output_tokens=500,
    )

    sql = response.output_text.strip()

    if not sql:
        raise RuntimeError("OpenAI returned an empty SQL query.")

    sql = re.sub(
        r"^```(?:sql)?\\s*",
        "",
        sql,
        flags=re.IGNORECASE,
    )
    sql = re.sub(r"\\s*```$", "", sql).strip()

    return sql

def validate_sql(sql):
    """Validate SQL and enforce an outer result limit."""

    if not sql or not sql.strip():
        raise ValueError("SQL query is empty.")

    statements = [
        statement
        for statement in sqlglot.parse(sql, read="postgres")
        if statement is not None
    ]

    if len(statements) != 1:
        raise ValueError("Only one SQL statement is allowed.")

    query = statements[0]

    # Only SELECT statements and SELECT-based UNION queries are allowed.
    if not isinstance(query, (exp.Select, exp.Union)):
        raise ValueError("Only SELECT queries are permitted.")

    # Reject SELECT INTO and row-locking clauses.
    if query.find(exp.Into) or query.find(exp.Lock):
        raise ValueError("SELECT INTO and locking queries are not permitted.")

    # Check every referenced table.
    cte_names = {
        cte.alias_or_name.lower()
        for cte in query.find_all(exp.CTE)
    }

    for table in query.find_all(exp.Table):
        table_name = table.name.lower()
        schema_name = (table.db or "").lower()

        if table_name not in ALLOWED_TABLES and table_name not in cte_names:
            raise ValueError(
                f"Table '{table_name}' is not approved for Copilot queries."
            )

        if schema_name and schema_name != "public":
            raise ValueError("Only the public database schema is permitted.")

        if table.catalog:
            raise ValueError("Cross-database queries are not permitted.")

    # Restrict function calls to a safe analytics allowlist.
    allowed_functions = {
        "COUNT",
        "SUM",
        "AVG",
        "MIN",
        "MAX",
        "COALESCE",
        "NULLIF",
        "LOWER",
        "UPPER",
        "TRIM",
        "ROUND",
        "ABS",
        "TIMESTAMP_TRUNC",
        "AND",
        "OR",
        "NOT",
    }

    for function in query.find_all(exp.Func):
        # CASE WHEN is a SQL conditional expression, not an unsafe function.
        if isinstance(function, (exp.Case, exp.If, exp.Cast)):
            continue

        if isinstance(function, exp.Anonymous):
            function_name = function.name.upper()
        else:
            function_name = function.sql_name().upper()

        if function_name not in allowed_functions:
            raise ValueError(
                f"SQL function '{function_name}' is not approved."
            )

    # Wrap the validated query to enforce a hard outer row limit.
    safe_sql = (
        "SELECT * FROM ("
        + query.sql(dialect="postgres")
        + ") AS copilot_result "
        + f"LIMIT {MAX_ROWS}"
    )

    return safe_sql


def execute_readonly_sql(engine, sql):
    """Execute validated SQL in a read-only PostgreSQL transaction."""

    safe_sql = validate_sql(sql)

    with engine.connect() as connection:
        with connection.begin():
            connection.exec_driver_sql(
                "SET TRANSACTION READ ONLY"
            )

            connection.exec_driver_sql(
                "SET LOCAL statement_timeout = '15s'"
            )

            result = pd.read_sql_query(
                text(safe_sql),
                connection,
            )

    return result


def ask_text_to_sql(engine, question):
    """Generate, validate and execute a natural-language SQL question."""

    generated_sql = generate_sql(question)

    safe_sql = validate_sql(generated_sql)

    with engine.connect() as connection:
        with connection.begin():
            connection.exec_driver_sql(
                "SET TRANSACTION READ ONLY"
            )

            connection.exec_driver_sql(
                "SET LOCAL statement_timeout = '15s'"
            )

            result = pd.read_sql_query(
                text(safe_sql),
                connection,
            )

    return {
        "question": question,
        "generated_sql": generated_sql,
        "executed_sql": safe_sql,
        "result": result,
        "row_count": len(result),
    }



def generate_template_sql(question):
    """Generate SQL from supported business questions without an LLM."""

    q = question.lower().strip()

    # Specific questions must be checked before generic customer-count questions.
    templates = [
        (
            [
                "customer acquisition channels",
                "customer acquisition channel distribution",
                "which channel brings the most customers",
                "which channels bring most customers",
                "customers by acquisition channel",
                "customer count by acquisition channel",
                "acquisition channel statistics",
                "top customer acquisition channel",
            ],
            """
            SELECT
                COALESCE(
                    NULLIF(TRIM(acquisition_channel), ''),
                    'Unknown'
                ) AS acquisition_channel,
                COUNT(DISTINCT customer_id) AS total_customers,
                ROUND(
                    100.0 * COUNT(DISTINCT customer_id) /
                    NULLIF(
                        SUM(COUNT(DISTINCT customer_id)) OVER (),
                        0
                    ),
                    2
                ) AS customer_percentage
            FROM customers
            GROUP BY 1
            ORDER BY total_customers DESC
            LIMIT 20
            """
        ),
        (
            [
                "monthly viewing activity",
                "monthly viewing trends",
                "monthly viewing statistics",
                "how many views each month",
                "how many views do we get each month",
                "monthly watch time",
                "show monthly watch time trends",
                "viewing activity by month",
                "monthly watch minutes",
            ],
            """
            SELECT
                DATE_TRUNC('month', viewing_date)::date AS viewing_month,
                COUNT(*) AS total_views,
                SUM(COALESCE(watch_duration_minutes, 0)) AS total_watch_minutes,
                COUNT(DISTINCT customer_id) AS active_viewers
            FROM viewing_activity
            WHERE viewing_date IS NOT NULL
            GROUP BY 1
            ORDER BY viewing_month
            """
        ),
        (
            [
                "average customer feedback rating",
                "average customer rating",
                "what is our average rating",
                "average feedback score",
                "customer satisfaction rating",
                "average rating given by customers",
                "how many feedback responses",
            ],
            """
            SELECT
                COUNT(rating) AS rated_feedback_count,
                ROUND(AVG(rating)::numeric, 2) AS average_rating,
                MIN(rating) AS minimum_rating,
                MAX(rating) AS maximum_rating
            FROM customer_feedback
            """
        ),
        (
            [
                "overall churn rate",
                "overall customer churn percentage",
                "what percentage of customers churned",
                "what is our churn rate",
                "customer churn percentage",
                "how many customers have churned",
            ],
            """
            SELECT
                COUNT(DISTINCT customer_id) AS total_customers,
                COUNT(DISTINCT customer_id) FILTER (
                    WHERE churned = TRUE
                ) AS churned_customers,
                ROUND(
                    100.0 * COUNT(DISTINCT customer_id) FILTER (
                        WHERE churned = TRUE
                    ) / NULLIF(COUNT(DISTINCT customer_id), 0),
                    2
                ) AS churn_rate_pct
            FROM churn_labels
            """
        ),
        (
            [
                "unresolved support tickets",
                "pending support tickets",
                "escalated support tickets",
                "how many support tickets are unresolved",
                "open support tickets",
                "support tickets by priority",
                "pending and escalated tickets",
            ],
            """
            SELECT
                ticket_status,
                priority,
                COUNT(*) AS ticket_count
            FROM support_tickets
            WHERE LOWER(TRIM(ticket_status)) IN ('pending', 'escalated')
            GROUP BY ticket_status, priority
            ORDER BY
                CASE
                    WHEN LOWER(TRIM(priority)) = 'critical' THEN 1
                    WHEN LOWER(TRIM(priority)) = 'high' THEN 2
                    WHEN LOWER(TRIM(priority)) = 'medium' THEN 3
                    WHEN LOWER(TRIM(priority)) = 'low' THEN 4
                    ELSE 5
                END,
                ticket_status
            """
        ),
        (
            [
                "popular viewing devices",
                "most popular devices",
                "which devices are used the most",
                "most used devices for watching",
                "viewing activity by device",
                "watch time by device",
                "device usage statistics",
            ],
            """
            SELECT
                CASE
                    WHEN LOWER(TRIM(device_type)) IN ('smart tv', 'smarttv', 'tv')
                        THEN 'Smart TV'
                    WHEN LOWER(TRIM(device_type)) IN ('mobile', 'smartphone', 'phone')
                        THEN 'Mobile'
                    WHEN LOWER(TRIM(device_type)) IN ('tablet', 'ipad')
                        THEN 'Tablet'
                    WHEN LOWER(TRIM(device_type)) IN ('laptop', 'desktop', 'computer', 'pc')
                        THEN 'Computer'
                    WHEN LOWER(TRIM(device_type)) IN (
                        'gaming console', 'console', 'playstation/xbox',
                        'playstation', 'xbox'
                    )
                        THEN 'Gaming Console'
                    ELSE COALESCE(NULLIF(TRIM(device_type), ''), 'Unknown')
                END AS device,
                COUNT(*) AS total_views,
                SUM(COALESCE(watch_duration_minutes, 0)) AS total_watch_minutes
            FROM viewing_activity
            GROUP BY 1
            ORDER BY total_views DESC
            LIMIT 10
            """
        ),
        (
            [
                "popular viewing devices",
                "most popular devices",
                "which devices are used the most",
                "most used devices for watching",
                "viewing activity by device",
                "watch time by device",
                "device usage statistics",
            ],
            """
            SELECT
                CASE
                    WHEN LOWER(TRIM(device_type)) IN ('smart tv', 'smarttv', 'tv')
                        THEN 'Smart TV'
                    WHEN LOWER(TRIM(device_type)) IN ('mobile', 'smartphone', 'phone')
                        THEN 'Mobile'
                    WHEN LOWER(TRIM(device_type)) IN ('tablet', 'ipad')
                        THEN 'Tablet'
                    WHEN LOWER(TRIM(device_type)) IN ('laptop', 'desktop', 'computer', 'pc')
                        THEN 'Computer'
                    ELSE COALESCE(NULLIF(TRIM(device_type), ''), 'Unknown')
                END AS device,
                COUNT(*) AS total_views,
                SUM(COALESCE(watch_duration_minutes, 0)) AS total_watch_minutes
            FROM viewing_activity
            GROUP BY 1
            ORDER BY total_views DESC
            LIMIT 10
            """
        ),
        (
            [
                "most watched genres",
                "most-watched genres",
                "which genres are watched the most",
                "most popular content genres",
                "top content genres",
                "genre with the highest watch time",
                "viewing time by genre",
                "total watch time by genre",
            ],
            """
            SELECT
                COALESCE(NULLIF(TRIM(c.genre), ''), 'Unknown') AS genre,
                COUNT(va.viewing_id) AS total_views,
                SUM(COALESCE(va.watch_duration_minutes, 0)) AS total_watch_minutes
            FROM viewing_activity va
            JOIN content c
                ON va.content_id = c.content_id
            GROUP BY 1
            ORDER BY total_watch_minutes DESC
            LIMIT 10
            """
        ),
        (
            [
                "failed payment rate",
                "percentage of failed payments",
                "what percentage of payments failed",
                "how many payments failed",
            ],
            """
            SELECT
                COUNT(*) FILTER (
                    WHERE payment_status = 'Failed'
                ) AS failed_payments,
                COUNT(*) AS total_payments,
                ROUND(
                    100.0 * COUNT(*) FILTER (
                        WHERE payment_status = 'Failed'
                    ) / NULLIF(COUNT(*), 0),
                    2
                ) AS failed_payment_rate
            FROM payments
            """
        ),
        (
            [
                "customers in each subscription plan",
                "how many customers are in each subscription plan",
                "how many customers in each subscription plan",
                "customer count by plan",
                "customers by subscription plan",
                "number of customers per plan",
            ],
            """
            SELECT
                sp.plan_name,
                COUNT(DISTINCT s.customer_id) AS customer_count
            FROM subscriptions s
            JOIN subscription_plans sp
                ON s.plan_id = sp.plan_id
            GROUP BY sp.plan_name
            ORDER BY customer_count DESC
            """
        ),
        (
            [
                "how many customers churned",
                "total churned customers",
                "number of churned customers",
                "count of churned customers",
            ],
            """
            SELECT COUNT(DISTINCT customer_id) AS churned_customers
            FROM churn_labels
            WHERE churned = TRUE
            """
        ),
        (
            [
                "most common churn reasons",
                "churn by reason",
                "churn reasons",
                "most common reason for churn",
            ],
            """
            SELECT
                churn_reason,
                COUNT(DISTINCT customer_id) AS churned_customers
            FROM churn_labels
            WHERE churned = TRUE
              AND churn_reason IS NOT NULL
            GROUP BY churn_reason
            ORDER BY churned_customers DESC
            """
        ),
        (
            [
                "countries have the highest",
                "which countries have the most customers",
                "which country has the most customers",
                "countries with the most customers",
                "which countries have most customers",
                "customers by country",
                "customer count by country",
                "top countries by customers",
                "most customers by country",
            ],
            """
            SELECT
                COALESCE(NULLIF(TRIM(country), ''), 'Unknown') AS country,
                COUNT(DISTINCT customer_id) AS customer_count
            FROM customers
            GROUP BY 1
            ORDER BY customer_count DESC
            LIMIT 5
            """
        ),
        (
            [
                "total successful payment revenue",
                "total successful revenue",
                "total successful payments",
                "total payment revenue",
            ],
            """
            SELECT SUM(amount) AS successful_payment_revenue
            FROM payments
            WHERE payment_status = 'Success'
            """
        ),
        (
            [
                "monthly successful payment",
                "monthly revenue trend",
                "monthly payment revenue",
            ],
            """
            SELECT
                DATE_TRUNC('month', payment_date) AS month,
                SUM(amount) AS successful_revenue
            FROM payments
            WHERE payment_status = 'Success'
            GROUP BY 1
            ORDER BY 1
            """
        ),
        (
            [
                "how many customers",
                "total number of customers",
                "total customers",
                "customer count",
            ],
            """
            SELECT COUNT(DISTINCT customer_id) AS total_customers
            FROM customers
            """
        ),
    ]

    for phrases, sql in templates:
        if any(phrase in q for phrase in phrases):
            return sql.strip()

    raise ValueError(
        "This question is not supported by offline SQL templates yet. "
        "Try asking about countries, payment revenue, subscription plans, "
        "customer count, or churn."
    )
