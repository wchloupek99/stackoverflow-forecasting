"""Cleaning, resampling, and feature engineering."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "tag_question_counts.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    """Load the raw tag question counts CSV."""
    df = pd.read_csv(path, parse_dates=["week"])
    df = df.drop_duplicates()
    return df


def resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample each tag to a consistent weekly frequency, filling gaps with 0."""
    full_weeks = pd.date_range(df["week"].min(), df["week"].max(), freq="W-MON")
    tags = df["tag"].unique()

    frames = []
    for tag in tags:
        tag_df = df[df["tag"] == tag].set_index("week")
        tag_df = tag_df.reindex(full_weeks)
        tag_df["tag"] = tag
        tag_df["question_count"] = tag_df["question_count"].fillna(0).astype(int)
        tag_df.index.name = "week"
        frames.append(tag_df.reset_index())

    return pd.concat(frames, ignore_index=True)


def to_prophet_format(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Convert to Prophet format (ds, y) — one DataFrame per tag.

    Returns a dict mapping tag name to its Prophet-ready DataFrame.
    """
    result = {}
    for tag, grp in df.groupby("tag"):
        prophet_df = pd.DataFrame({
            "ds": grp["week"].values,
            "y": grp["question_count"].values,
        })
        prophet_df = prophet_df.sort_values("ds").reset_index(drop=True)
        result[tag] = prophet_df
    return result


def save_processed(tag_dfs: dict[str, pd.DataFrame], output_dir: Path = PROCESSED_DIR):
    """Save each tag's Prophet-ready DataFrame as a CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for tag, df in tag_dfs.items():
        filename = f"{tag.replace('.', '_')}.csv"
        df.to_csv(output_dir / filename, index=False)
        print(f"  Saved {filename} ({len(df)} rows)")


if __name__ == "__main__":
    print("Loading raw data...")
    raw = load_raw()
    print(f"  {len(raw)} rows, {raw['tag'].nunique()} tags")

    print("Resampling to consistent weekly frequency...")
    resampled = resample_weekly(raw)
    filled_count = len(resampled) - len(raw)
    if filled_count > 0:
        print(f"  Filled {filled_count} missing week(s) with 0")

    print("Converting to Prophet format (ds, y)...")
    tag_dfs = to_prophet_format(resampled)

    print(f"Saving to {PROCESSED_DIR}/")
    save_processed(tag_dfs)

    print("Done.")
