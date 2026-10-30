from pathlib import Path
import re
import shutil

path = Path("sql/sql_business_questions.sql")
sql = path.read_text(encoding="utf-8")

backup = path.with_name("sql_business_questions_backup_q31_q33.sql")
shutil.copy2(path, backup)

# ---------------- Q31: Fix numeric normalization ----------------
q31_start = sql.index("-- Q31:")
q32_start = sql.index("-- Q32:", q31_start)
q31 = sql[q31_start:q32_start]

old_recency = "(days_inactive - MIN(days_inactive) OVER ())"
new_recency = "(days_inactive - MIN(days_inactive) OVER ())::NUMERIC"

old_support = "(ticket_count - MIN(ticket_count) OVER ())"
new_support = "(ticket_count - MIN(ticket_count) OVER ())::NUMERIC"

if old_recency not in q31 or old_support not in q31:
    raise ValueError("Q31 expected normalization expressions not found. No changes saved.")

q31 = q31.replace(old_recency, new_recency, 1)
q31 = q31.replace(old_support, new_support, 1)

# ---------------- Q33: Restrict engagement to before churn ----------------
q33_start = sql.index("-- Q33:")
q34_start = sql.index("-- Q34:", q33_start)
q33 = sql[q33_start:q34_start]

old_filter = "WHERE cl.churned = TRUE"
new_filter = """WHERE cl.churned = TRUE
      AND cl.churn_date IS NOT NULL"""

if old_filter not in q33:
    raise ValueError("Q33 churn filter not found. No changes saved.")

q33 = q33.replace(old_filter, new_filter, 1)

old_engagement = """last_engagement AS (
    SELECT
        customer_id,
        MAX(viewing_date) AS last_viewing_date,
        SUM(watch_duration_minutes) AS total_watch_minutes
    FROM viewing_activity
    GROUP BY customer_id
)"""

new_engagement = """last_engagement AS (
    SELECT
        va.customer_id,
        MAX(va.viewing_date) AS last_viewing_date,
        SUM(va.watch_duration_minutes) AS total_watch_minutes
    FROM viewing_activity va
    JOIN churned_value cv
        ON cv.customer_id = va.customer_id
    WHERE va.viewing_date <= cv.churn_date
    GROUP BY va.customer_id
)"""

if old_engagement not in q33:
    raise ValueError("Q33 last_engagement CTE not found. No changes saved.")

q33 = q33.replace(old_engagement, new_engagement, 1)

# Reassemble file
sql = sql[:q31_start] + q31 + sql[q32_start:q33_start] + q33 + sql[q34_start:]

path.write_text(sql, encoding="utf-8")

print("Q31: Numeric casts added to recency and support normalization.")
print("Q33: Post-churn viewing excluded.")
print("Q33: Customers with NULL churn_date excluded because pre-churn CLV cannot be established.")
print(f"Backup created: {backup}")
print("SQL file updated successfully.")
