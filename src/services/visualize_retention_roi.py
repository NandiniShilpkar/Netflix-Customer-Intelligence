from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

INPUT = Path("reports/retention_roi/retention_sensitivity_results.csv")
OUTPUT = Path("reports/retention_roi/visualizations")
OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

sns.set_theme(style="whitegrid")

# 1. ROI heatmap for each campaign cost and revenue period.
fig, axes = plt.subplots(3, 4, figsize=(19, 13))

for i, months in enumerate(sorted(df["revenue_saving_months"].unique())):
    for j, cost in enumerate(sorted(df["campaign_cost_per_customer"].unique())):
        ax = axes[i, j]

        subset = df[
            (df["revenue_saving_months"] == months)
            & (df["campaign_cost_per_customer"] == cost)
        ]

        pivot = subset.pivot(
            index="retention_success_rate",
            columns="contact_capacity_pct",
            values="estimated_roi_pct",
        )

        sns.heatmap(
            pivot,
            annot=True,
            fmt=".1f",
            cmap="RdYlGn",
            center=0,
            ax=ax,
            cbar=(i == 0 and j == 3),
        )

        ax.set_title(f"Cost: {cost:.2f} | Revenue: {months} months")
        ax.set_xlabel("Contact capacity (%)")
        ax.set_ylabel("Retention success rate")

fig.suptitle("Retention Campaign ROI Sensitivity Heatmaps", fontsize=17)
plt.tight_layout()
fig.savefig(OUTPUT / "roi_sensitivity_heatmaps.png", dpi=180)
plt.close(fig)

# 2. Average ROI by capacity and retention rate.
summary = (
    df.groupby(
        ["contact_capacity_pct", "retention_success_rate"],
        as_index=False,
    )["estimated_roi_pct"].mean()
)

plt.figure(figsize=(11, 6))

sns.lineplot(
    data=summary,
    x="retention_success_rate",
    y="estimated_roi_pct",
    hue="contact_capacity_pct",
    marker="o",
)

plt.axhline(0, linestyle="--")
plt.title("Average ROI by Retention Success Rate")
plt.xlabel("Assumed retention success rate")
plt.ylabel("Average estimated ROI (%)")
plt.tight_layout()
plt.savefig(OUTPUT / "average_roi_comparison.png", dpi=180)
plt.close()

# 3. Profitable scenario distribution.
profit_summary = (
    df.groupby("contact_capacity_pct")["profitable"]
    .agg(["sum", "count"])
    .reset_index()
)

profit_summary["unprofitable"] = (
    profit_summary["count"] - profit_summary["sum"]
)

profit_summary = profit_summary.rename(
    columns={
        "sum": "Profitable",
        "unprofitable": "Unprofitable",
    }
)

profit_summary.set_index("contact_capacity_pct")[
    ["Profitable", "Unprofitable"]
].plot(
    kind="bar",
    stacked=True,
    figsize=(10, 6),
)

plt.title("Profitable vs Unprofitable Scenarios")
plt.xlabel("Contact capacity (%)")
plt.ylabel("Number of scenarios")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT / "scenario_profitability.png", dpi=180)
plt.close()

print("VISUALIZATION COMPLETED")
for file in sorted(OUTPUT.glob("*.png")):
    print(file)
