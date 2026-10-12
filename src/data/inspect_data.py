from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw")
REPORT_DIR = Path("data/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(DATA_DIR.glob("*.csv"))

if not files:
    raise FileNotFoundError(
        "No CSV files found in data/raw/. Check your file locations."
    )

summary = []

for file in files:
    df = pd.read_csv(file, low_memory=False)

    print("\n" + "=" * 60)
    print(f"FILE: {file.name}")
    print(f"ROWS: {len(df):,}")
    print(f"COLUMNS: {len(df.columns)}")

    print("\nCOLUMN NAMES:")
    print(df.columns.tolist())

    print("\nDATA TYPES:")
    print(df.dtypes.astype(str).to_string())

    print("\nMISSING VALUES:")
    missing = df.isna().sum()
    print(missing[missing > 0].to_string() or "No missing values")

    print("\nDUPLICATE ROWS:", df.duplicated().sum())

    print("\nFIRST 3 ROWS:")
    print(df.head(3).to_string(index=False))

    for column in df.columns:
        summary.append({
            "file": file.name,
            "rows": len(df),
            "columns": len(df.columns),
            "column": column,
            "dtype": str(df[column].dtype),
            "missing_count": int(df[column].isna().sum()),
            "missing_pct": round(
                df[column].isna().mean() * 100, 2
            ),
            "distinct_values": int(df[column].nunique(dropna=True)),
        })

report = pd.DataFrame(summary)
output_file = REPORT_DIR / "data_quality_initial.csv"
report.to_csv(output_file, index=False)

print("\n" + "=" * 60)
print("DATA INSPECTION COMPLETED")
print(f"CSV FILES INSPECTED: {len(files)}")
print(f"REPORT SAVED TO: {output_file}")
