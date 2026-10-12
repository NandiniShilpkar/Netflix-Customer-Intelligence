from pathlib import Path
import pandas as pd

DATA = Path("data/processed")

def load(name):
    return pd.read_csv(DATA / name, low_memory=False)

customers = load("customers.csv")
subscriptions = load("subscriptions.csv")
payments = load("payments.csv")
plans = load("subscription_plans.csv")
content = load("content.csv")
viewing = load("viewing_activity.csv")
feedback = load("customer_feedback.csv")
tickets = load("support_tickets.csv")
churn = load("churn_labels.csv")

checks = []

def check(label, source, column, target, target_column):
    source_ids = source[column].dropna()
    target_ids = set(target[target_column].dropna())
    missing = source_ids[~source_ids.isin(target_ids)]

    checks.append({
        "relationship": label,
        "source_rows": len(source_ids),
        "unmatched_rows": len(missing),
        "unmatched_unique_ids": missing.nunique(),
    })

check("subscriptions -> customers", subscriptions, "customer_id",
      customers, "customer_id")
check("subscriptions -> plans", subscriptions, "plan_id",
      plans, "plan_id")
check("payments -> customers", payments, "customer_id",
      customers, "customer_id")
check("viewing -> customers", viewing, "customer_id",
      customers, "customer_id")
check("viewing -> content", viewing, "content_id",
      content, "content_id")
check("feedback -> customers", feedback, "customer_id",
      customers, "customer_id")
check("tickets -> customers", tickets, "customer_id",
      customers, "customer_id")
check("churn labels -> customers", churn, "customer_id",
      customers, "customer_id")

result = pd.DataFrame(checks)
Path("data/reports").mkdir(parents=True, exist_ok=True)
result.to_csv("data/reports/relationship_validation.csv", index=False)

print("\nRELATIONSHIP VALIDATION")
print(result.to_string(index=False))

print("\nPRIMARY ID DUPLICATE CHECKS")
for name, df, col in [
    ("customers", customers, "customer_id"),
    ("subscriptions", subscriptions, "subscription_id"),
    ("payments", payments, "payment_id"),
    ("plans", plans, "plan_id"),
    ("content", content, "content_id"),
    ("viewing", viewing, "viewing_id"),
    ("feedback", feedback, "feedback_id"),
    ("tickets", tickets, "ticket_id"),
]:
    duplicates = df[col].dropna().duplicated().sum()
    print(f"{name}: {duplicates} duplicate IDs")

print("\nReport saved to data/reports/relationship_validation.csv")
