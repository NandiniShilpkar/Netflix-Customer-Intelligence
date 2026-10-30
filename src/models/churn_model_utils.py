import pandas as pd


def prepare_temporal_split(
    df,
    target,
    excluded_columns,
    train_dates,
    test_dates,
):
    """Validate historical churn data and create a chronological split."""

    required = set(excluded_columns + ["customer_id"])
    missing_columns = required - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df.copy()

    df["snapshot_date"] = pd.to_datetime(
        df["snapshot_date"], errors="raise"
    )
    df["outcome_window_end"] = pd.to_datetime(
        df["outcome_window_end"], errors="raise"
    )

    if df[target].isna().any():
        raise ValueError("Target contains missing values.")

    if not set(df[target].unique()).issubset({0, 1}):
        raise ValueError("Target must contain only 0 and 1.")

    if df.duplicated(["customer_id", "snapshot_date"]).any():
        raise ValueError("Duplicate customer-snapshot pairs found.")

    train_df = df[
        df["snapshot_date"].dt.strftime("%Y-%m-%d").isin(train_dates)
    ].copy()

    test_df = df[
        df["snapshot_date"].dt.strftime("%Y-%m-%d").isin(test_dates)
    ].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Training or testing dataset is empty.")

    if train_df["snapshot_date"].max() >= test_df["snapshot_date"].min():
        raise ValueError("Training dates must precede testing dates.")

    if train_df["outcome_window_end"].max() >= test_df["snapshot_date"].min():
        raise ValueError(
            "Training outcome windows overlap the test period."
        )

    if train_df[target].nunique() != 2 or test_df[target].nunique() != 2:
        raise ValueError(
            "Both training and testing data must contain both target classes."
        )

    return train_df, test_df
