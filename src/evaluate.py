"""Metrics and plots for forecast evaluation."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


def compute_metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    """Compute MAE, RMSE, and MAPE."""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def plot_forecast(actual: pd.Series, predicted: pd.Series, title: str = "Forecast vs Actual", save_path: str | None = None):
    """Plot actual vs predicted values."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(actual.index, actual.values, label="Actual")
    ax.plot(predicted.index, predicted.values, label="Predicted")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.show()
    return fig
