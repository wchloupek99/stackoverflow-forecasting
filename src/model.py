"""Prophet model training and forecasting for each tag."""

from pathlib import Path

import pandas as pd
from prophet import Prophet

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FORECAST_DIR = PROJECT_ROOT / "outputs" / "forecasts"

FORECAST_WEEKS = 52


def load_processed(tag: str, processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Load a single tag's processed Prophet-ready CSV."""
    filename = f"{tag.replace('.', '_')}.csv"
    return pd.read_csv(processed_dir / filename, parse_dates=["ds"])


def get_available_tags(processed_dir: Path = PROCESSED_DIR) -> list[str]:
    """Return tag names derived from CSVs in the processed directory."""
    return [p.stem for p in sorted(processed_dir.glob("*.csv"))]


def fit_prophet(df: pd.DataFrame) -> Prophet:
    """Fit a Prophet model with yearly and weekly seasonality."""
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
    )
    model.fit(df)
    return model


def forecast(model: Prophet, periods: int = FORECAST_WEEKS) -> pd.DataFrame:
    """Generate a forecast DataFrame for the given number of weeks."""
    future = model.make_future_dataframe(periods=periods, freq="W-MON")
    return model.predict(future)


def save_forecast(forecast_df: pd.DataFrame, tag: str, output_dir: Path = FORECAST_DIR):
    """Save forecast results to a CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cols = ["ds", "yhat", "yhat_lower", "yhat_upper", "trend", "yearly", "weekly"]
    cols = [c for c in cols if c in forecast_df.columns]
    filepath = output_dir / f"{tag}_forecast.csv"
    forecast_df[cols].to_csv(filepath, index=False)
    return filepath


if __name__ == "__main__":
    tags = get_available_tags()
    print(f"Found {len(tags)} tag(s): {tags}\n")

    for tag in tags:
        print(f"[{tag}] Loading processed data...")
        df = load_processed(tag)
        print(f"  {len(df)} weeks of history")

        print(f"[{tag}] Fitting Prophet model...")
        model = fit_prophet(df)

        print(f"[{tag}] Generating {FORECAST_WEEKS}-week forecast...")
        forecast_df = forecast(model)

        path = save_forecast(forecast_df, tag)
        print(f"[{tag}] Saved to {path}\n")

    print("Done — all forecasts saved.")
