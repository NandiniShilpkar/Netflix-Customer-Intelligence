from pathlib import Path
import nbformat as nbf

notebooks = Path("notebooks")
notebooks.mkdir(exist_ok=True)

def make_notebook(cells, filename):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    }
    nbf.write(nb, notebooks / filename)
    print("Created:", notebooks / filename)

# --------------------------------------------------
# 01 - DATA CLEANING
# --------------------------------------------------

clean_cells = [
    nbf.v4.new_markdown_cell("""# 01 — Data Cleaning and Quality Validation

## Project
Netflix Customer Intelligence & AI Analytics Copilot

## Objective
Inspect and clean the nine source datasets before analytics and modelling.

This notebook documents and executes the existing cleaning implementation in
`src/data/clean_data.py`. It does not replace or silently modify that pipeline.

## Cleaning operations
- Normalize surrounding whitespace in text columns.
- Convert empty strings to missing values.
- Remove exact duplicate rows from the customer dataset only.
- Mark customer ages outside 0–120 as missing.
- Parse configured date columns.
- Report duplicate identifiers without automatically deleting them.
- Validate selected numeric ranges.
- Save processed CSV files and a cleaning report.

**Important:** Numeric validation findings are reported; the pipeline does not
automatically remove or correct every invalid numeric value.
"""),
    nbf.v4.new_markdown_cell("## 1. Imports and project paths"),
    nbf.v4.new_code_cell("""from pathlib import Path
import pandas as pd
from IPython.display import display

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "src" / "data" / "clean_data.py").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "data" / "reports"

print("Project root:", PROJECT_ROOT)
print("Raw data directory exists:", RAW_DIR.exists())
print("Raw CSV count:", len(list(RAW_DIR.glob("*.csv"))))
"""),
    nbf.v4.new_markdown_cell("## 2. Inspect raw datasets"),
    nbf.v4.new_code_cell("""raw_files = sorted(RAW_DIR.glob("*.csv"))
if len(raw_files) != 9:
    raise ValueError(f"Expected 9 raw CSV files, found {len(raw_files)}")

raw_summary = []
for file in raw_files:
    df = pd.read_csv(file, low_memory=False)
    raw_summary.append({
        "file": file.name,
        "rows": len(df),
        "columns": len(df.columns),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum())
    })

raw_summary = pd.DataFrame(raw_summary)
display(raw_summary)
"""),
    nbf.v4.new_markdown_cell("""## 3. Execute the project cleaning pipeline

The next cell runs the existing source script. It writes cleaned datasets to
`data/processed/` and the report to `data/reports/cleaning_report.csv`.
"""),
    nbf.v4.new_code_cell("""%run ../src/data/clean_data.py"""),
    nbf.v4.new_markdown_cell("## 4. Review the cleaning report"),
    nbf.v4.new_code_cell("""cleaning_report_path = REPORT_DIR / "cleaning_report.csv"
cleaning_report = pd.read_csv(cleaning_report_path)
display(cleaning_report)
"""),
    nbf.v4.new_markdown_cell("## 5. Compare raw and processed datasets"),
    nbf.v4.new_code_cell("""comparison = []
for file in raw_files:
    raw = pd.read_csv(file, low_memory=False)
    cleaned = pd.read_csv(PROCESSED_DIR / file.name, low_memory=False)
    comparison.append({
        "file": file.name,
        "raw_rows": len(raw),
        "processed_rows": len(cleaned),
        "rows_removed": len(raw) - len(cleaned),
        "raw_columns": len(raw.columns),
        "processed_columns": len(cleaned.columns),
        "processed_missing_cells": int(cleaned.isna().sum().sum())
    })

comparison = pd.DataFrame(comparison)
display(comparison)
"""),
    nbf.v4.new_markdown_cell("""## 6. Data-cleaning conclusions and limitations

- The pipeline processes all nine expected datasets.
- Exact duplicate removal is deliberately restricted to customer records.
- Duplicate identifiers in other datasets are reported for investigation.
- Implausible ages are marked missing rather than imputed.
- Malformed dates are converted to `NaT`.
- Numeric validation checks report issues but do not universally repair them.
- Relationship integrity is assessed separately by the project's relationship
  validation script.

The generated cleaning report should be reviewed before downstream modelling.
""")
]
make_notebook(clean_cells, "01_Data_Cleaning.ipynb")

# --------------------------------------------------
# 03 - FEATURE ENGINEERING
# --------------------------------------------------

feature_cells = [
    nbf.v4.new_markdown_cell("""# 03 — Historical Churn Feature Engineering

## Project
Netflix Customer Intelligence & AI Analytics Copilot

## Objective
Construct point-in-time customer snapshots for predicting churn during the
following 30 days.

The notebook executes the existing scripts:
- `src/features/build_churn_features.py`
- `src/features/build_advanced_churn_features.py`

The historical cutoffs are January–May 2026. Base features use viewing activity
from the preceding 90 days and payment/support history through each cutoff.
Advanced features compare the latest 30 days with the preceding 30 days.
"""),
    nbf.v4.new_markdown_cell("## 1. Imports and project paths"),
    nbf.v4.new_code_cell("""from pathlib import Path
import pandas as pd
from IPython.display import display

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "src" / "features" / "build_churn_features.py").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

FEATURE_DIR = PROJECT_ROOT / "data" / "features"
print("Project root:", PROJECT_ROOT)
print("Feature directory:", FEATURE_DIR)
"""),
    nbf.v4.new_markdown_cell("## 2. Build historical snapshots"),
    nbf.v4.new_code_cell("""%run ../src/features/build_churn_features.py"""),
    nbf.v4.new_markdown_cell("## 3. Build advanced behavioural features"),
    nbf.v4.new_code_cell("""%run ../src/features/build_advanced_churn_features.py"""),
    nbf.v4.new_markdown_cell("## 4. Load and inspect engineered dataset"),
    nbf.v4.new_code_cell("""base_path = FEATURE_DIR / "churn_historical_snapshots.csv"
advanced_path = FEATURE_DIR / "churn_advanced_snapshots.csv"

base = pd.read_csv(base_path)
features = pd.read_csv(advanced_path)

print("Base shape:", base.shape)
print("Advanced shape:", features.shape)
print("Unique customers:", features["customer_id"].nunique())
print("Duplicate customer-snapshot pairs:",
      features.duplicated(["customer_id", "snapshot_date"]).sum())

display(features.head())
"""),
    nbf.v4.new_markdown_cell("## 5. Target distribution and snapshot coverage"),
    nbf.v4.new_code_cell("""target_summary = (
    features.groupby("snapshot_date")["churn_next_30_days"]
    .agg(rows="size", churned="sum", churn_rate="mean")
    .reset_index()
)
display(target_summary)

print("\\nOverall target distribution:")
display(features["churn_next_30_days"].value_counts(dropna=False))
"""),
    nbf.v4.new_markdown_cell("## 6. Feature groups"),
    nbf.v4.new_code_cell("""feature_groups = {
    "Customer profile": [
        "age", "gender", "country", "city", "preferred_language"
    ],
    "Viewing behaviour": [
        "viewing_sessions_90d", "active_viewing_days_90d",
        "total_watch_minutes_90d", "avg_watch_minutes_90d",
        "avg_completion_pct_90d", "days_since_last_viewing",
        "viewing_sessions_30d", "watch_minutes_30d",
        "watch_minutes_previous_30d", "watch_minutes_change_pct",
        "viewing_sessions_change_pct", "active_days_change_pct"
    ],
    "Payment behaviour": [
        "payment_count", "failed_payment_count",
        "successful_payment_count", "failed_payment_rate",
        "payments_30d", "failed_payments_30d",
        "failed_payment_rate_30d", "payment_failures_change"
    ],
    "Support behaviour": [
        "support_ticket_count", "high_priority_ticket_count",
        "unresolved_ticket_count", "avg_support_satisfaction",
        "avg_resolution_hours", "support_tickets_30d",
        "unresolved_tickets_30d", "escalated_tickets_30d"
    ],
    "Risk indicators": [
        "declining_engagement_flag", "payment_risk_flag",
        "support_risk_flag", "has_unresolved_support_30d",
        "has_critical_support_30d"
    ]
}

for group, columns in feature_groups.items():
    present = [column for column in columns if column in features.columns]
    print(f"{group}: {len(present)} available columns")
    print(", ".join(present))
"""),
    nbf.v4.new_markdown_cell("""## 7. Data-quality and leakage checks

The source pipeline uses historical date filters for viewing and payment
events. However, support-ticket status fields represent current recorded
statuses. If those statuses were updated after a snapshot cutoff, the resulting
support features may contain information unavailable at prediction time.

This is a leakage risk that requires investigation before claiming that the
model is fully point-in-time safe.

The 30-day churn target also requires complete follow-up observation. The
source script labels snapshots as provisional and explicitly asks for this
verification.
"""),
    nbf.v4.new_code_cell("""required = ["customer_id", "snapshot_date", "churn_next_30_days"]
missing_required = [c for c in required if c not in features.columns]
if missing_required:
    raise ValueError(f"Missing required columns: {missing_required}")

checks = {
    "duplicate_customer_snapshot_pairs":
        int(features.duplicated(["customer_id", "snapshot_date"]).sum()),
    "missing_customer_ids": int(features["customer_id"].isna().sum()),
    "missing_snapshot_dates": int(features["snapshot_date"].isna().sum()),
    "invalid_target_values":
        int((~features["churn_next_30_days"].isin([0, 1])).sum())
}
display(pd.Series(checks, name="count").to_frame())
"""),
    nbf.v4.new_markdown_cell("""## 8. Feature-engineering conclusions

The project creates historical customer snapshots and adds behavioural
features for engagement decline, payment friction and support risk.

The engineered dataset is suitable for further model experimentation, subject
to:
1. Verifying full 30-day outcome follow-up for every snapshot.
2. Investigating support-status temporal leakage.
3. Keeping customer identifiers and snapshot metadata out of model predictors.
4. Performing model validation that accounts for repeated customers across
   historical snapshots.

Feature importance and model performance indicate predictive association,
not causal explanations of churn.
""")
]
make_notebook(feature_cells, "03_Feature_Engineering.ipynb")

print("\\nBoth notebooks created.")
