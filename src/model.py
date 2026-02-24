"""Prophet / ARIMA model training and forecasting."""

import pandas as pd


def train_prophet(df: pd.DataFrame, date_col: str = "ds", value_col: str = "y", periods: int = 30):
    """Train a Prophet model and return the model with forecast."""
    from prophet import Prophet

    train_df = df.rename(columns={date_col: "ds", value_col: "y"})
    model = Prophet()
    model.fit(train_df)

    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return model, forecast


def train_arima(series: pd.Series, order: tuple = (1, 1, 1), steps: int = 30):
    """Train an ARIMA model and return the forecast."""
    from statsmodels.tsa.arima.model import ARIMA

    model = ARIMA(series, order=order)
    fitted = model.fit()
    forecast = fitted.forecast(steps=steps)
    return fitted, forecast
