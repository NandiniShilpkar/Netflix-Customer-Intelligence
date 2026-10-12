from pathlib import Path
import pandas as pd

RAW = Path("data/raw")

def show(title, df, columns=None):
    print(f"\n{'=' * 65}\n{title}")
    if df.empty:
        print("No records found.")
    else:
        print(df[columns].head(15).to_string(index=False)
              if columns else df.head(15).to_string(index=False))

# 1. Duplicate customer IDs
customers = pd.read_csv(RAW / "customers.csv")
duplicate_mask = customers["customer_id"].duplicated(keep=False)
show(
    "DUPLICATE CUSTOMER RECORDS",
    customers.loc[duplicate_mask].sort_values("customer_id"),
    ["customer_id", "first_name", "last_name",
     "registration_date", "country", "age"]
)

# 2. Invalid ages
ages = pd.to_numeric(customers["age"], errors="coerce")
show(
    "AGES OUTSIDE 0-120",
    customers.loc[ages.notna() & ~ages.between(0, 120)],
    ["customer_id", "age"]
)

# 3. Support ticket dates that failed parsing
tickets = pd.read_csv(RAW / "support_tickets.csv")
ticket_dates = pd.to_datetime(tickets["ticket_date"], errors="coerce")
show(
    "INVALID SUPPORT TICKET DATES",
    tickets.loc[tickets["ticket_date"].notna() & ticket_dates.isna()],
    ["ticket_id", "ticket_date", "ticket_status"]
)

# 4. Satisfaction scores: separate missing from truly invalid
scores = pd.to_numeric(
    tickets["customer_satisfaction_score"], errors="coerce"
)
print("\nSATISFACTION SCORE CHECK")
print("Missing:", int(scores.isna().sum()))
print("Non-missing outside 1-5:",
      int((scores.notna() & ~scores.between(1, 5)).sum()))
print("Observed score values:")
print(scores.value_counts(dropna=False).sort_index().to_string())

# 5. Completion percentage: separate missing from invalid
viewing = pd.read_csv(RAW / "viewing_activity.csv")
completion = pd.to_numeric(
    viewing["completion_percentage"], errors="coerce"
)
print("\nCOMPLETION PERCENTAGE CHECK")
print("Missing:", int(completion.isna().sum()))
print("Non-missing outside 0-100:",
      int((completion.notna() & ~completion.between(0, 100)).sum()))

# 6. Churn labels and missing-value patterns
churn = pd.read_csv(RAW / "churn_labels.csv")
print("\nCHURN LABEL COUNTS")
print(churn["churned"].value_counts(dropna=False).to_string())
print("\nMISSING CHURN FIELDS BY LABEL")
print(churn.groupby("churned", dropna=False)[
    ["churn_date", "churn_reason"]
].agg(lambda s: s.isna().sum()).to_string())

# 7. Payment missingness by status
payments = pd.read_csv(RAW / "payments.csv")
print("\nPAYMENT STATUS COUNTS")
print(payments["payment_status"].value_counts(dropna=False).to_string())
print("\nMISSING FAILURE REASONS BY STATUS")
print(payments.groupby("payment_status", dropna=False)[
    "failed_payment_reason"
].agg(
    total="size",
    missing=lambda s: s.isna().sum()
).to_string())

# 8. Confirm source files remain intact
print("\nDIAGNOSTICS COMPLETED. Raw files were not modified.")
