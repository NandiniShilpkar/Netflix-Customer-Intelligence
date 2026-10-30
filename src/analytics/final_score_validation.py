import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
engine = create_engine(os.environ["DATABASE_URL"])

queries = {
"DATA_CUTOFF": """
SELECT
    CURRENT_DATE AS system_date,
    (SELECT MAX(viewing_date)::date FROM viewing_activity) AS latest_viewing_date,
    (SELECT MAX(payment_date)::date FROM payments) AS latest_payment_date,
    (SELECT MAX(registration_date)::date FROM customers) AS latest_registration_date;
""",

"Q31_SCORE_VALIDATION": """
WITH latest_subscription AS (
    SELECT s.*,
        ROW_NUMBER() OVER (
            PARTITION BY customer_id
            ORDER BY subscription_start_date DESC, subscription_id DESC
        ) AS rn
    FROM subscriptions s
    WHERE subscription_status = 'Active'
      AND subscription_end_date IS NULL
),
engagement AS (
    SELECT customer_id, MAX(viewing_date) AS last_activity_date
    FROM viewing_activity GROUP BY customer_id
),
payment_metrics AS (
    SELECT customer_id,
        COUNT(*) FILTER (WHERE payment_status = 'Failed') AS failed_count,
        COUNT(*) AS total_count
    FROM payments GROUP BY customer_id
),
tickets AS (
    SELECT customer_id, COUNT(*) AS ticket_count
    FROM support_tickets GROUP BY customer_id
),
base AS (
    SELECT s.customer_id, s.monthly_price,
        COALESCE(CURRENT_DATE - e.last_activity_date, 365) AS days_inactive,
        COALESCE(p.failed_count::numeric / NULLIF(p.total_count, 0), 0)
            AS failed_payment_rate,
        COALESCE(t.ticket_count, 0) AS ticket_count
    FROM latest_subscription s
    LEFT JOIN engagement e ON e.customer_id = s.customer_id
    LEFT JOIN payment_metrics p ON p.customer_id = s.customer_id
    LEFT JOIN tickets t ON t.customer_id = s.customer_id
    WHERE s.rn = 1
),
normalized AS (
    SELECT *,
        COALESCE(
            (days_inactive - MIN(days_inactive) OVER ())::numeric /
            NULLIF(MAX(days_inactive) OVER () - MIN(days_inactive) OVER (), 0), 0
        ) AS recency_score,
        COALESCE(
            (failed_payment_rate - MIN(failed_payment_rate) OVER ()) /
            NULLIF(MAX(failed_payment_rate) OVER () - MIN(failed_payment_rate) OVER (), 0), 0
        ) AS payment_score,
        COALESCE(
            (ticket_count - MIN(ticket_count) OVER ())::numeric /
            NULLIF(MAX(ticket_count) OVER () - MIN(ticket_count) OVER (), 0), 0
        ) AS support_score
    FROM base
),
scored AS (
    SELECT *,
        (recency_score * 0.4 + payment_score * 0.35 +
         support_score * 0.25) * 100 AS risk_score
    FROM normalized
)
SELECT
    COUNT(*) AS customers_scored,
    MIN(risk_score) AS minimum_score,
    MAX(risk_score) AS maximum_score,
    COUNT(*) FILTER (WHERE risk_score < 0 OR risk_score > 100) AS out_of_range,
    COUNT(*) FILTER (WHERE risk_score IS NULL) AS null_scores,
    MIN(recency_score) AS min_recency,
    MAX(recency_score) AS max_recency,
    MIN(payment_score) AS min_payment,
    MAX(payment_score) AS max_payment,
    MIN(support_score) AS min_support,
    MAX(support_score) AS max_support
FROM scored;
""",

"Q37_SCORE_VALIDATION": """
WITH latest_subscription AS (
    SELECT s.customer_id, s.plan_id,
        ROW_NUMBER() OVER (
            PARTITION BY s.customer_id
            ORDER BY s.subscription_start_date DESC, s.subscription_id DESC
        ) AS rn
    FROM subscriptions s
    WHERE s.subscription_status = 'Active'
      AND s.subscription_end_date IS NULL
),
engagement AS (
    SELECT customer_id, MAX(viewing_date) AS last_viewing_date
    FROM viewing_activity GROUP BY customer_id
),
payment_metrics AS (
    SELECT customer_id, COUNT(*) AS total_payments,
        COUNT(*) FILTER (WHERE payment_status = 'Failed') AS failed_payments
    FROM payments GROUP BY customer_id
),
support_metrics AS (
    SELECT customer_id, AVG(customer_satisfaction_score) AS avg_satisfaction
    FROM support_tickets
    WHERE customer_satisfaction_score IS NOT NULL
    GROUP BY customer_id
),
base AS (
    SELECT c.customer_id,
        COALESCE(CURRENT_DATE - e.last_viewing_date, 365) AS inactive_days,
        COALESCE(pm.failed_payments::numeric / NULLIF(pm.total_payments, 0), 0)
            AS payment_failure_rate,
        COALESCE(sm.avg_satisfaction, 3) AS avg_satisfaction,
        COALESCE(CURRENT_DATE - c.registration_date, 0) AS tenure_days
    FROM customers c
    LEFT JOIN latest_subscription ls ON ls.customer_id = c.customer_id AND ls.rn = 1
    LEFT JOIN engagement e ON e.customer_id = c.customer_id
    LEFT JOIN payment_metrics pm ON pm.customer_id = c.customer_id
    LEFT JOIN support_metrics sm ON sm.customer_id = c.customer_id
),
normalized AS (
    SELECT *,
        COALESCE((inactive_days - MIN(inactive_days) OVER ())::numeric /
            NULLIF(MAX(inactive_days) OVER () - MIN(inactive_days) OVER (), 0), 0)
            AS recency_risk,
        COALESCE((payment_failure_rate - MIN(payment_failure_rate) OVER ()) /
            NULLIF(MAX(payment_failure_rate) OVER () - MIN(payment_failure_rate) OVER (), 0), 0)
            AS payment_risk,
        COALESCE((MAX(avg_satisfaction) OVER () - avg_satisfaction) /
            NULLIF(MAX(avg_satisfaction) OVER () - MIN(avg_satisfaction) OVER (), 0), 0)
            AS support_risk,
        COALESCE((MAX(tenure_days) OVER () - tenure_days)::numeric /
            NULLIF(MAX(tenure_days) OVER () - MIN(tenure_days) OVER (), 0), 0)
            AS tenure_risk
    FROM base
),
scored AS (
    SELECT *,
        (recency_risk * 0.35 + payment_risk * 0.30 +
         support_risk * 0.20 + tenure_risk * 0.15) * 100 AS risk_score
    FROM normalized
)
SELECT
    COUNT(*) AS customers_scored,
    MIN(risk_score) AS minimum_score,
    MAX(risk_score) AS maximum_score,
    COUNT(*) FILTER (WHERE risk_score < 0 OR risk_score > 100) AS out_of_range,
    COUNT(*) FILTER (WHERE risk_score IS NULL) AS null_scores,
    MIN(recency_risk) AS min_recency,
    MAX(recency_risk) AS max_recency,
    MIN(payment_risk) AS min_payment,
    MAX(payment_risk) AS max_payment,
    MIN(support_risk) AS min_support,
    MAX(support_risk) AS max_support,
    MIN(tenure_risk) AS min_tenure,
    MAX(tenure_risk) AS max_tenure
FROM scored;
""",

"Q35_WEEK_BOUNDARIES": """
WITH latest AS (
    SELECT MAX(viewing_date)::date AS max_date FROM viewing_activity
),
weekly AS (
    SELECT
        DATE_TRUNC('week', viewing_date)::date AS week_start,
        COUNT(DISTINCT customer_id) AS wau
    FROM viewing_activity, latest
    WHERE viewing_date >= DATE_TRUNC('month', latest.max_date) - INTERVAL '5 months'
      AND viewing_date <= latest.max_date
    GROUP BY 1
)
SELECT
    DATE_TRUNC('month', week_start)::date AS assigned_month,
    COUNT(*) AS weekly_buckets,
    COUNT(*) FILTER (
        WHERE week_start < DATE_TRUNC('month', week_start)::date
           OR week_start + 6 > (DATE_TRUNC('month', week_start)
                                + INTERVAL '1 month' - INTERVAL '1 day')::date
    ) AS weeks_crossing_month_boundary,
    MIN(week_start) AS first_week_start,
    MAX(week_start) AS last_week_start,
    ROUND(AVG(wau), 2) AS average_wau
FROM weekly
GROUP BY 1
ORDER BY 1;
"""
}

out = ROOT / "data/reports/sql_question_results"
out.mkdir(parents=True, exist_ok=True)

for name, query in queries.items():
    print("\n" + "=" * 65)
    print(name)
    print("=" * 65)
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        print(df.to_string(index=False))
        df.to_csv(out / f"{name.lower()}.csv", index=False)
    except Exception as exc:
        print("ERROR:", exc)

print("\nValidation reports saved in:", out)
