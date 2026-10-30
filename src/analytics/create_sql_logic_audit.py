from pathlib import Path
import csv
import re

ROOT = Path(__file__).resolve().parents[2]
SQL_FILE = ROOT / "sql/sql_business_questions.sql"
OUT_DIR = ROOT / "data/reports/sql_question_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sql = SQL_FILE.read_text(encoding="utf-8")

pattern = re.compile(r"(?im)^\s*--\s*(Q\d+)\s*:\s*(.*)$")
matches = list(pattern.finditer(sql))

queries = {}
for i, match in enumerate(matches):
    qid = match.group(1).upper()
    end = matches[i + 1].start() if i + 1 < len(matches) else len(sql)
    queries[qid] = sql[match.end():end]


# Audit status and observations.
audit = {
    "Q05": (
        "WARNING",
        "Historical plan memberships may count one customer under multiple plans. "
        "Interpret as historical membership, not exclusive current-plan population."
    ),
    "Q06": (
        "REVIEW",
        "Confirm watch-duration aggregation and title ranking against viewing_activity grain."
    ),
    "Q07": (
        "REVIEW",
        "Confirm completion percentage is averaged at session level and grouped by content type and genre."
    ),
    "Q08": (
        "REVIEW",
        "Confirm the 12-month window includes the intended calendar months."
    ),
    "Q09": (
        "REVIEW",
        "Check missing resolution times and whether averages exclude unresolved tickets."
    ),
    "Q10": (
        "REVIEW",
        "Confirm only resolved/closed tickets with valid satisfaction scores are included."
    ),
    "Q11": (
        "VALIDATED",
        "Device labels normalized into standard categories. "
        "All 66,422 viewing rows have valid watch durations."
    ),
    "Q13": (
        "VALIDATED",
        "All 8,000 customers have unique IDs and valid registration dates. "
        "Eight acquisition channels were found."
    ),
    "Q15": (
        "VALIDATED",
        "Ticket aggregation validated: 6,120 tickets across 4,408 customers. "
        "Exactly 120 customers have more than three tickets; maximum is seven."
    ),
    "Q11": (
        "REVIEW",
        "Confirm viewing minutes are grouped by device type without unintended duplicate joins."
    ),
    "Q12": (
        "WARNING",
        "Feedback sentiment labels contain known intentional mislabelling. "
        "Treat sentiment averages cautiously."
    ),
    "Q13": (
        "REVIEW",
        "Confirm registration counts use distinct customers and calendar-year grouping."
    ),
    "Q14": (
        "REVIEW",
        "Verify auto-renew is measured from the latest subscription per customer."
    ),
    "Q15": (
        "REVIEW",
        "Confirm ticket counts are customer-level and strictly greater than three."
    ),
    "Q16": (
        "REVIEW",
        "Active subscription revenue is an MRR proxy. Check for overlapping active subscription records."
    ),
    "Q17": (
        "REVIEW",
        "Most recent ticket per customer should use ROW_NUMBER with deterministic tie-breaking."
    ),
    "Q18": (
        "VALIDATED",
        "Calendar-month diagnostic found no missing registration months in this dataset. "
        "The three-row moving average has consecutive observed calendar months for this dataset."
    ),
    "Q19": (
        "VALIDATED",
        "Corrected to compare the latest viewing month with the exact previous calendar month. "
        "Nonconsecutive observed months are no longer treated as consecutive."
    ),
    "Q20": (
        "VALIDATED",
        "Corrected to exclude customers whose 1-, 3- or 6-month retention milestone "
        "falls after the observation cutoff."
    ),
    "Q21": (
        "REVIEW",
        "Check country denominators and small-country sample sizes when ranking churn rates."
    ),
    "Q22": (
        "REVIEW",
        "Running totals should include successful payments only and use deterministic payment ordering."
    ),
    "Q23": (
        "REVIEW",
        "Confirm three declining months are consecutive calendar months."
    ),
    "Q24": (
        "REVIEW",
        "Verify churn numerator includes only customers active at the start of each month."
    ),
    "Q25": (
        "REVIEW",
        "Confirm current retention uses one latest subscription record per customer."
    ),
    "Q26": (
        "REVIEW",
        "Lifetime spend should include successful payments only; quartiles should cover the intended population."
    ),
    "Q27": (
        "REVIEW",
        "Confirm ROW_NUMBER ranks subcategories within each country and handles ties deterministically."
    ),
    "Q28": (
        "WARNING",
        "Failed-payment risk is a rule-based proxy, not a trained ML churn probability."
    ),
    "Q29": (
        "DATA LIMITATION",
        "No historical subscription price changes were found. Zero rows are an expected "
        "dataset limitation, not an execution failure."
    ),
    "Q30": (
        "VALIDATED",
        "Dataset validation found zero NULL churn_date values among churned customers. "
        "The NULL churn-date concern does not occur in this dataset."
    ),
    "Q31": (
        "VALIDATED WITH CAVEAT",
        "Numeric normalization checks passed: 6,009 customers scored, no NULL or "
        "out-of-range scores. Uses CURRENT_DATE (system date 2026-10-30) while "
        "latest viewing data is 2026-06-30. Score is a rule-based risk proxy."
    ),
    "Q32": (
        "REVIEW",
        "Confirm the reporting window is the most recent completed quarter relative to dataset dates."
    ),
    "Q33": (
        "VALIDATED",
        "Corrected to restrict viewing activity to dates on or before churn_date "
        "before calculating last engagement."
    ),
    "Q34": (
        "VALIDATED",
        "Corrected to distinguish migration events from unique customers and calculate "
        "churn rate using distinct customers."
    ),
    "Q35": (
        "WARNING",
        "Weekly buckets are assigned to the month containing their Monday start date. "
        "Some weeks cross month boundaries; document this convention when interpreting WAU."
    ),
    "Q36": (
        "WARNING",
        "Negative sentiment labels are imperfect. Combining them with low ratings "
        "helps but does not eliminate label noise."
    ),
    "Q37": (
        "VALIDATED WITH CAVEAT",
        "Score validation passed for 8,000 customers: no NULL or out-of-range scores "
        "and all normalized components are between 0 and 1. Uses CURRENT_DATE "
        "(2026-10-30) while latest viewing data is 2026-06-30. Feature weights "
        "are heuristic and predictive performance has not yet been established."
    ),
    "Q38": (
        "REVIEW",
        "Verify plan assignment at churn date for churned customers and latest subscription for non-churned customers."
    ),
}



# Evidence-based updates from Q06-Q17 database diagnostics.
audit.update({
    "Q01": (
        "VALIDATED",
        "The latest subscription is selected per customer using ROW_NUMBER "
        "with subscription_start_date and subscription_id ordering. "
        "The result covers 6,009 active customers across 21 normalized countries."
    ),
    "Q02": (
        "VALIDATED WITH CAVEAT",
        "All 103,529 successful payments were mapped to subscriptions using "
        "customer ID and effective subscription date boundaries. The average "
        "is average payment amount, not average monthly revenue per customer."
    ),
    "Q03": (
        "VALIDATED",
        "The churn_labels table contains 8,000 unique customer records: "
        "1,991 churned and 6,009 non-churned. Overall churn rate is 24.89%."
    ),
    "Q04": (
        "VALIDATED",
        "Acquisition channel analysis covers 8,000 unique customers across "
        "8 channels. Distinct customer counts and churn-rate calculations "
        "were verified."
    ),
})

audit.update({
    "Q06": (
        "VALIDATED",
        "All 66,422 viewing rows have matching content records. "
        "No NULL or unmatched content IDs were found."
    ),
    "Q07": (
        "VALIDATED",
        "65,891 non-NULL completion values were found, ranging from 1 to 100. "
        "No out-of-range values or unmatched content records were found."
    ),
    "Q08": (
        "VALIDATED",
        "The latest payment month is May 2026. The 12-month reporting window "
        "contains failed payments in all 12 months, from June 2025 through May 2026."
    ),
    "Q09": (
        "VALIDATED",
        "All 4,592 Resolved/Closed tickets have resolution times. "
        "No negative resolution times were found."
    ),
    "Q10": (
        "VALIDATED",
        "All 4,592 Resolved/Closed tickets have non-NULL satisfaction scores "
        "within the observed 1-5 range."
    ),
    "Q11": (
        "VALIDATED",
        "Device labels normalized into standard categories. "
        "All 66,422 viewing rows have valid watch durations."
    ),
    "Q13": (
        "VALIDATED",
        "All 8,000 customers have unique IDs and valid registration dates. "
        "Eight acquisition channels were found."
    ),
    "Q15": (
        "VALIDATED",
        "Ticket aggregation validated: 6,120 tickets across 4,408 customers. "
        "Exactly 120 customers have more than three tickets; maximum is seven."
    ),
    "Q14": (
        "VALIDATED",
        "Latest-subscription ranking returns exactly 8,000 rows for 8,000 unique customers."
    ),
    "Q16": (
        "VALIDATED",
        "There are 6,009 active subscription rows for 6,009 unique customers. "
        "No overlapping active subscriptions were found."
    ),
    "Q17": (
        "VALIDATED WITH CAVEAT",
        "ROW_NUMBER uses ticket_date DESC NULLS LAST and ticket_id as a tie-breaker. "
        "However, 61 tickets have missing dates and 30 customers have only undated tickets. "
        "Their selected ticket cannot be verified as the latest by date."
    ),

    "Q21": (
        "VALIDATED",
        "All 8,000 customers have country mappings across 21 normalized categories. "
        "Country churn rates and DENSE_RANK ordering were verified."
    ),
    "Q22": (
        "VALIDATED",
        "All 103,529 successful payments have unique payment IDs, valid customer IDs, "
        "payment dates and amounts. No negative amounts or duplicate ordering keys."
    ),
    "Q23": (
        "VALIDATED",
        "The query checks consecutive calendar months using month-date comparisons. "
        "It identified 1,447 declining three-month windows across 1,205 customers."
    ),
    "Q24": (
        "VALIDATED WITH CAVEAT",
        "Historical activity is reconstructed using subscription start and cancellation "
        "dates with exclusive end boundaries. All 1,991 churn dates match cancellation "
        "dates. Forty-two records with cancellation dates before subscription start "
        "are excluded from the denominator."
    ),
    "Q25": (
        "VALIDATED",
        "Latest-subscription ranking returns 8,000 customer records: 6,009 Active "
        "and 1,991 Cancelled. No latest Active record has a non-NULL end date."
    ),
    "Q26": (
        "VALIDATED",
        "Lifetime successful spend was calculated for all 8,000 customers. NTILE(4) "
        "produces four equal groups of 2,000 customers each."
    ),
    "Q27": (
        "VALIDATED",
        "All 6,120 tickets have non-NULL, non-blank subcategories. Ranking returns "
        "exactly three subcategories for each of 21 countries, totaling 63 rows."
    ),


    "Q30": (
        "VALIDATED WITH CAVEAT",
        "All 8,000 customers are represented and all 1,991 churned customers "
        "have churn dates and reasons. Successful payments after churn were "
        "found for 339 churned customers; the CLV query correctly excludes "
        "payments after the churn date."
    ),
    "Q32": (
        "VALIDATED",
        "The most recent completed quarter is January-March 2026. It contains "
        "586 tickets across 547 customers. All have subcategories, and the "
        "ranking returns exactly five complaints for each of 21 countries."
    ),
    "Q38": (
        "VALIDATED WITH CAVEAT",
        "All 8,000 customers receive exactly one plan assignment, including "
        "all 1,991 churned customers at their churn dates. No tied latest "
        "subscription start dates were found. Forty-two customers have "
        "churn dates before the start of another subscription record."
    ),

})

def get_section(qid):
    raw = queries.get(qid, "")
    lines = raw.splitlines()

    start = next(
        (
            i for i, line in enumerate(lines)
            if re.match(r"^\s*(SELECT|WITH)\b", line, re.I)
        ),
        None
    )

    return "\n".join(lines[start:]) if start is not None else ""


# Basic static safety checks.
q31 = get_section("Q31").lower()

if q31 and "::numeric" not in q31 and "::decimal" not in q31:
    audit["Q31"] = (
        "REVIEW",
        "No explicit NUMERIC/DECIMAL cast found in Q31. Recheck normalization "
        "formulas before relying on the earlier validation."
    )

q33 = get_section("Q33").lower()

if q33 and "churn_date" not in q33:
    audit["Q33"] = (
        "REVIEW",
        "Q33 does not reference churn_date. Recheck the last-engagement cutoff."
    )


# Load SQL execution summary.
summary_path = OUT_DIR / "execution_summary.csv"
execution = {}

if summary_path.exists():
    with summary_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            execution[row["question_id"].upper()] = row


# Write final business logic audit.
output = OUT_DIR / "business_logic_validation.csv"

with output.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "question_id",
        "execution_status",
        "result_rows",
        "logic_audit_status",
        "audit_observation"
    ])

    for n in range(1, 39):
        qid = f"Q{n:02d}"

        logic_status, observation = audit.get(
            qid,
            ("NOT REVIEWED", "No audit note available.")
        )

        run = execution.get(qid, {})

        writer.writerow([
            qid,
            run.get("status", "NOT FOUND"),
            run.get("result_rows", ""),
            logic_status,
            observation
        ])


# Print report summary.
status_counts = {}

for status, _ in audit.values():
    status_counts[status] = status_counts.get(status, 0) + 1

print("=" * 65)
print("SQL BUSINESS LOGIC AUDIT REPORT")
print("=" * 65)

print(f"Questions audited: {len(audit)}")
print(f"SQL questions detected: {len(queries)}")
print(f"Execution summary found: {summary_path.exists()}")

print("\nLOGIC AUDIT STATUS COUNTS")
for status, count in sorted(status_counts.items()):
    print(f"{status}: {count}")

print(f"\nReport saved: {output}")

print("\nIMPORTANT:")
print("This report combines static SQL checks with previously completed validations.")
print("It is not proof that every business calculation is correct.")
print("Q31 and Q37 use CURRENT_DATE and have a historical data-cutoff caveat.")
print("Q31 and Q37 are heuristic risk scores, not validated ML probabilities.")
print("=" * 65)
