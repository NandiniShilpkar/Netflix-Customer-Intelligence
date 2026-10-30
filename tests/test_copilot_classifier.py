import pytest

from src.dashboard.ai_analytics_copilot import classify_question


@pytest.mark.parametrize(
    "question, expected",
    [
        ("Show data quality score", "data_quality_mlops"),
        ("Explain the top churn drivers", "data_quality_mlops"),
        ("Show model monitoring", "production_ai_monitoring"),
        ("Is the model production ready?", "production_ai_monitoring"),
        ("Give me an executive summary", "executive_analytics"),
        ("What should leadership focus on?", "executive_analytics"),
        ("Explain our retention strategy", "retention_optimization"),
        ("How much revenue can we recover?", "retention_optimization"),
        ("How much revenue is at risk?", "revenue_at_risk"),
        ("Show customer health score", "customer_intelligence"),
        ("Which customers should we save first?", "customer_intelligence"),
        ("Show retention targets", "retention_targets"),
        ("What should the business do about high-risk customers?", "retention_actions"),
        ("Why are customers leaving?", "churn_reasons"),
        ("Which customer segment has the highest churn?", "churn_segments"),
        ("Which subscription plan has the highest churn rate?", "churn_by_plan"),
        ("Which customers are most likely to churn?", "highest_churn"),
        ("Explain artificial intelligence history", "unknown"),
    ],
)
def test_question_classification(question, expected):
    assert classify_question(question) == expected


@pytest.mark.parametrize(
    "question, expected",
    [
        ("Show customer profile for CUST123456", "customer_360:CUST123456"),
        ("What is churn risk of CUST654321?", "customer_360:CUST654321"),
        ("Show customer 360 for cust123456", "customer_360:CUST123456"),
    ],
)
def test_customer_360_classification(question, expected):
    assert classify_question(question) == expected


def test_data_quality_phrase_priority():
    assert classify_question("Show data quality") == "data_quality_mlops"


def test_general_subscription_question_uses_sql_fallback():
    assert classify_question("Which subscription plan has the most customers?") == "unknown"
