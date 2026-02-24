# Stack Overflow Forecasting

Forecasting weekly question volume on Stack Overflow by tag using time series models. The project extracts historical data from Google BigQuery's public Stack Overflow dataset, explores trends and seasonality through interactive EDA, then fits per-tag Prophet models to generate 52-week forecasts with backtesting evaluation.

## Data

**Source:** [`bigquery-public-data.stackoverflow`](https://console.cloud.google.com/bigquery?p=bigquery-public-data&d=stackoverflow) on Google Cloud.

The query selects the top 10 tags by all-time question count (e.g. javascript, python, java, c#, etc.) and aggregates weekly question counts from 2009 to present.

| Stage | Location | Description |
|---|---|---|
| Raw | `data/raw/tag_question_counts.csv` | Weekly question counts per tag straight from BigQuery |
| Processed | `data/processed/{tag}.csv` | One CSV per tag in Prophet format (`ds`, `y`), gap-filled to a consistent weekly frequency |

## Methods & Frameworks

- **BigQuery** (via `google-cloud-bigquery`) for data extraction from the public SO dataset
- **Pandas** for data wrangling, resampling, and gap-filling
- **Plotly** for interactive EDA charts
- **Prophet** for time series modeling with yearly and weekly seasonality
- **scikit-learn** + **matplotlib** for evaluation metrics (MAE, RMSE, MAPE) and train/test plots

## EDA Notebook

[`notebooks/eda.ipynb`](notebooks/eda.ipynb) loads the raw data and performs:

- Per-tag summary statistics (mean, std, min, median, max)
- Interactive line charts of weekly question volume (combined and per-tag facets)
- Missing weeks detection across all tags
- Outlier flagging via rolling z-score (13-week window, 3-sigma threshold)
- Box plot distributions of weekly counts by tag

## Pipeline

Run each script from the project root in order:

```
# 1. Extract data from BigQuery
python src/query.py

# 2. Clean, resample, and convert to Prophet format
python src/preprocess.py

# 3. Fit Prophet models and generate 52-week forecasts
python src/model.py

# 4. Backtest evaluation (hold out last 52 weeks)
python src/evaluate.py
```

| Script | What it does |
|---|---|
| `src/query.py` | Authenticates with GCP, queries the top-10-tag weekly counts, saves raw CSV |
| `src/preprocess.py` | Loads the raw CSV, fills missing weeks with zeros, outputs per-tag CSVs in `ds`/`y` format |
| `src/model.py` | Loads each processed CSV, fits Prophet (yearly + weekly seasonality), saves 52-week forecast CSVs |
| `src/evaluate.py` | Holds out the last 52 weeks, fits Prophet on training data, computes MAE/RMSE/MAPE, saves metrics CSV and per-tag evaluation charts |

## Output

All outputs are written to `outputs/forecasts/`:

| File | Description |
|---|---|
| `{tag}_forecast.csv` | Full forecast with `yhat`, confidence bounds, trend, and seasonality components |
| `{tag}_eval.png` | Actual vs predicted chart for the 52-week holdout period |
| `evaluation_metrics.csv` | MAE, RMSE, and MAPE for every tag in one table |

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

3. Place your GCP service account key at `credentials/gcp_key.json`. The `.env` file points `GOOGLE_APPLICATION_CREDENTIALS` to this path.
