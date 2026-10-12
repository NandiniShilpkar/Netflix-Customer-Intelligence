from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
REPORT_DIR = Path("data/reports")

OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DATE_COLUMNS = {
    "customers.csv": ["registration_date"],
    "subscriptions.csv": [
        "subscription_start_date",
        "subscription_end_date",
        "cancellation_date",
    ],
    "payments.csv": ["payment_date"],
    "support_tickets.csv": ["ticket_date"],
    "customer_feedback.csv": ["feedback_date"],
    "viewing_activity.csv": ["viewing_date"],
    "churn_labels.csv": ["churn_date"],
}

ID_COLUMNS = {
    "customers.csv": ["customer_id"],
    "subscriptions.csv": ["subscription_id"],
    "payments.csv": ["payment_id"],
    "support_tickets.csv": ["ticket_id"],
    "customer_feedback.csv": ["feedback_id"],
    "viewing_activity.csv": ["viewing_id"],
    "content.csv": ["content_id"],
    "subscription_plans.csv": ["plan_id"],
    "churn_labels.csv": ["customer_id"],
}

reports = []
files = sorted(RAW_DIR.glob("*.csv"))

if len(files) != 9:
    raise ValueError(f"Expected 9 CSV files; found {len(files)}.")

for file in files:
    df = pd.read_csv(file, low_memory=False)
    rows_before = len(df)

    # Normalize surrounding whitespace in text fields.
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].map(
            lambda x: x.strip() if isinstance(x, str) else x
        )
        df[col] = df[col].replace("", pd.NA)

    # Remove exact duplicate customer records only.
    exact_duplicates_removed = 0
    if file.name == "customers.csv":
        exact_duplicates_removed = int(df.duplicated().sum())
        df = df.drop_duplicates().copy()

    # Mark implausible ages as missing; do not invent replacements.
    invalid_ages = 0
    if file.name == "customers.csv" and "age" in df.columns:
        age = pd.to_numeric(df["age"], errors="coerce")
        invalid_mask = age.notna() & ~age.between(0, 120)
        invalid_ages = int(invalid_mask.sum())
        df["age"] = age.mask(invalid_mask)

    # Parse dates; malformed nonblank dates become NaT.
    invalid_dates = {}
    for col in DATE_COLUMNS.get(file.name, []):
        if col in df.columns:
            original = df[col].copy()
            parsed = pd.to_datetime(original, errors="coerce")
            invalid_dates[col] = int(
                (original.notna() & parsed.isna()).sum()
            )
            df[col] = parsed

    # Duplicate IDs are reported, not automatically removed.
    duplicate_ids = {}
    for col in ID_COLUMNS.get(file.name, []):
        if col in df.columns:
            duplicate_ids[col] = int(
                df[col].notna().sum() - df[col].nunique(dropna=True)
            )

    # Validate only observed, non-missing numeric values.
    checks = {}

    def count_outside(column, lower, upper):
        values = pd.to_numeric(df[column], errors="coerce")
        return int(
            (values.notna() & ~values.between(lower, upper)).sum()
        )

    if file.name == "customer_feedback.csv":
        checks["rating_outside_1_5"] = count_outside("rating", 1, 5)

    if file.name == "support_tickets.csv":
        checks["satisfaction_outside_1_5"] = count_outside(
            "customer_satisfaction_score", 1, 5
        )
        values = pd.to_numeric(
            df["resolution_time_hours"], errors="coerce"
        )
        checks["negative_resolution_time"] = int(
            (values.notna() & (values < 0)).sum()
        )

    if file.name == "viewing_activity.csv":
        checks["completion_outside_0_100"] = count_outside(
            "completion_percentage", 0, 100
        )
        for col in [
            "watch_duration_minutes",
            "session_duration_minutes",
        ]:
            values = pd.to_numeric(df[col], errors="coerce")
            checks[f"negative_{col}"] = int(
                (values.notna() & (values < 0)).sum()
            )

    if file.name == "payments.csv":
        values = pd.to_numeric(df["amount"], errors="coerce")
        checks["negative_payment_amount"] = int(
            (values.notna() & (values < 0)).sum()
        )

    output_path = OUT_DIR / file.name
    df.to_csv(output_path, index=False, date_format="%Y-%m-%d")

    reports.append({
        "file": file.name,
        "rows_before": rows_before,
        "rows_after": len(df),
        "rows_removed": rows_before - len(df),
        "exact_duplicates_removed": exact_duplicates_removed,
        "invalid_ages_marked_missing": invalid_ages,
        "invalid_dates_by_column": str(invalid_dates),
        "duplicate_ids": str(duplicate_ids),
        "validation_checks": str(checks),
        "missing_cells_after": int(df.isna().sum().sum()),
    })

report = pd.DataFrame(reports)
report.to_csv(REPORT_DIR / "cleaning_report.csv", index=False)

print("CLEANING PIPELINE COMPLETED")
print(f"Files processed: {len(files)}")
print(f"Cleaned data: {OUT_DIR}")
print(f"Cleaning report: {REPORT_DIR / 'cleaning_report.csv'}")
print(report.to_string(index=False))
