import pytest

from src.services.text_to_sql import (
    generate_template_sql,
    validate_sql,
)


SUPPORTED_QUESTIONS = [
    "What is the failed payment rate?",
    "Which genres are watched the most?",
    "Which devices are used the most?",
    "How many support tickets are unresolved?",
    "What is our overall churn rate?",
    "What is our average customer rating?",
    "Show monthly viewing activity",
    "Which channel brings the most customers?",
]


@pytest.mark.parametrize("question", SUPPORTED_QUESTIONS)
def test_offline_template_generation(question):
    sql = generate_template_sql(question)

    assert isinstance(sql, str)
    assert sql.strip()


@pytest.mark.parametrize("question", SUPPORTED_QUESTIONS)
def test_offline_template_sql_validation(question):
    sql = generate_template_sql(question)

    validate_sql(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM customers",
        "DROP TABLE customers",
        "UPDATE customers SET age = 30",
        "SELECT * FROM unauthorized_table",
        "SELECT 1; SELECT 2",
        "SELECT pg_sleep(10)",
    ],
)
def test_unsafe_sql_is_rejected(sql):
    with pytest.raises(Exception):
        validate_sql(sql)


def test_unsupported_question_has_no_template():
    with pytest.raises(ValueError):
        generate_template_sql(
            "Explain the history of artificial intelligence"
        )
