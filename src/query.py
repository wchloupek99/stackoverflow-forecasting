"""BigQuery extraction logic for Stack Overflow data."""

from google.cloud import bigquery
from dotenv import load_dotenv

load_dotenv()


def get_client() -> bigquery.Client:
    """Return an authenticated BigQuery client."""
    return bigquery.Client()


def run_query(query: str, client: bigquery.Client | None = None) -> "pd.DataFrame":
    """Execute a BigQuery SQL query and return results as a DataFrame."""
    import pandas as pd

    if client is None:
        client = get_client()
    return client.query(query).to_dataframe()
