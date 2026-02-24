"""Cleaning, resampling, and feature engineering."""

import pandas as pd


def load_raw(path: str) -> pd.DataFrame:
    """Load a raw data file (CSV or Parquet)."""
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, parse_dates=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop duplicates and handle missing values."""
    df = df.drop_duplicates()
    df = df.dropna()
    return df


def resample_daily(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    """Resample a time-indexed DataFrame to daily frequency."""
    df = df.set_index(pd.to_datetime(df[date_col]))
    return df[[value_col]].resample("D").sum().reset_index()
