# Stack Overflow Forecasting

Time series forecasting on Stack Overflow data using Google BigQuery.

## Setup

1. Create and activate a virtual environment:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Place your GCP service account key at `credentials/gcp_key.json`.

## Usage

- **`src/query.py`** — Extract data from BigQuery
- **`src/preprocess.py`** — Clean and resample data
- **`src/model.py`** — Train Prophet or ARIMA models
- **`src/evaluate.py`** — Compute metrics and plot forecasts
- **`notebooks/eda.ipynb`** — Exploratory data analysis
