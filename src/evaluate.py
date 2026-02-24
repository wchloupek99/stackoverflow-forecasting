"""Train/test evaluation: hold out last 52 weeks, fit Prophet, measure accuracy."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from model import load_processed, get_available_tags, fit_prophet

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FORECAST_DIR = PROJECT_ROOT / "outputs" / "forecasts"

HOLDOUT_WEEKS = 52  # 12 months ≈ 52 weeks


def train_test_split(df: pd.DataFrame, holdout: int = HOLDOUT_WEEKS):
    """Split a Prophet-formatted DataFrame into train and test sets."""
    cutoff = len(df) - holdout
    return df.iloc[:cutoff].copy(), df.iloc[cutoff:].copy()


def predict_test_period(model, test_df: pd.DataFrame) -> pd.DataFrame:
    """Generate predictions for the test period dates."""
    future = test_df[["ds"]].copy()
    return model.predict(future)


def compute_metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    """Compute MAE, RMSE, and MAPE."""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual.replace(0, np.nan))) * 100
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE": round(mape, 2)}


def plot_actual_vs_predicted(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    tag: str,
    save_path: Path | None = None,
):
    """Plot training data, actual test data, and predicted values."""
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(train_df["ds"], train_df["y"], color="steelblue", label="Train")
    ax.plot(test_df["ds"], test_df["y"], color="darkorange", label="Actual (test)")
    ax.plot(forecast_df["ds"], forecast_df["yhat"], color="green", linestyle="--", label="Predicted")
    ax.fill_between(
        forecast_df["ds"],
        forecast_df["yhat_lower"],
        forecast_df["yhat_upper"],
        color="green",
        alpha=0.1,
        label="Confidence interval",
    )

    ax.axvline(test_df["ds"].iloc[0], color="gray", linestyle=":", alpha=0.7)
    ax.set_title(f"{tag} — Actual vs Predicted (last {HOLDOUT_WEEKS} weeks held out)")
    ax.set_xlabel("Week")
    ax.set_ylabel("Question Count")
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


if __name__ == "__main__":
    FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    tags = get_available_tags()
    print(f"Evaluating {len(tags)} tag(s): {tags}\n")

    all_metrics = []

    for tag in tags:
        print(f"[{tag}] Loading processed data...")
        df = load_processed(tag)

        train_df, test_df = train_test_split(df)
        print(f"  Train: {len(train_df)} weeks | Test: {len(test_df)} weeks")

        print(f"[{tag}] Fitting Prophet on training set...")
        model = fit_prophet(train_df)

        print(f"[{tag}] Predicting test period...")
        pred_df = predict_test_period(model, test_df)

        metrics = compute_metrics(test_df["y"], pred_df["yhat"])
        metrics["tag"] = tag
        all_metrics.append(metrics)
        print(f"  MAE={metrics['MAE']}  RMSE={metrics['RMSE']}  MAPE={metrics['MAPE']}%")

        chart_path = FORECAST_DIR / f"{tag}_eval.png"
        plot_actual_vs_predicted(train_df, test_df, pred_df, tag, save_path=chart_path)
        print(f"  Chart saved to {chart_path}\n")

    metrics_df = pd.DataFrame(all_metrics)[["tag", "MAE", "RMSE", "MAPE"]]
    metrics_path = FORECAST_DIR / "evaluation_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(metrics_df.to_string(index=False))
    print(f"\nMetrics saved to {metrics_path}")
