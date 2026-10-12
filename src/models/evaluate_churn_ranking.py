
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "features" / "churn_features.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

TARGET = "churn_next_30_days"
RANDOM_STATE = 42
TEST_SIZE = 0.20

df = pd.read_csv(DATA_PATH)

# Reproduce the same test split used during training.
from sklearn.model_selection import train_test_split

X = df.drop(columns=[TARGET, "customer_id"])
y = df[TARGET].astype(int)

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

models = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.joblib",
    "Random Forest": MODEL_DIR / "random_forest.joblib",
}

rows = []

for model_name, model_path in models.items():
    model = joblib.load(model_path)
    risk_scores = model.predict_proba(X_test)[:, 1]

    ranked = pd.DataFrame(
        {
            "actual_churn": y_test.to_numpy(),
            "risk_score": risk_scores,
        }
    ).sort_values("risk_score", ascending=False)

    total_churners = int(ranked["actual_churn"].sum())

    print("\n" + "=" * 55)
    print(model_name)
    print("=" * 55)
    print(f"Test customers: {len(ranked):,}")
    print(f"Actual churners: {total_churners}")
    print(
        "Average precision:",
        round(average_precision_score(y_test, risk_scores), 4),
    )

    for fraction in (0.05, 0.10, 0.20):
        n = max(1, int(len(ranked) * fraction))
        top = ranked.head(n)
        captured = int(top["actual_churn"].sum())

        precision = float(top["actual_churn"].mean())
        recall = (
            captured / total_churners
            if total_churners
            else 0.0
        )

        print(
            f"Top {fraction:.0%}: "
            f"{n} customers | "
            f"{captured} churners | "
            f"precision={precision:.3f} | "
            f"recall={recall:.3f}"
        )

        rows.append(
            {
                "model": model_name,
                "top_fraction": fraction,
                "customers_flagged": n,
                "churners_captured": captured,
                "precision": precision,
                "recall": recall,
            }
        )

output_path = REPORT_DIR / "churn_ranking_metrics.csv"
pd.DataFrame(rows).to_csv(output_path, index=False)

print(f"\nRanking results saved to: {output_path}")
