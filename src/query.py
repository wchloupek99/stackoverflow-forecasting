"""BigQuery extraction logic for Stack Overflow data."""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

TOP_TAGS_QUERY = """
WITH top_tags AS (
    SELECT tag
    FROM `bigquery-public-data.stackoverflow.tags`
    ORDER BY count DESC
    LIMIT 10
)
SELECT
    DATE_TRUNC(q.creation_date, WEEK(MONDAY)) AS week,
    tag,
    COUNT(*) AS question_count
FROM
    `bigquery-public-data.stackoverflow.posts_questions` q,
    UNNEST(SPLIT(q.tags, '|')) AS tag
WHERE
    tag IN (SELECT tag FROM top_tags)
    AND q.creation_date >= '2009-01-01'
GROUP BY
    week, tag
ORDER BY
    week, tag
"""


def get_client() -> bigquery.Client:
    """Return an authenticated BigQuery client."""
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(
            PROJECT_ROOT / credentials_path
        )
    return bigquery.Client()


def run_query(query: str, client: bigquery.Client | None = None) -> pd.DataFrame:
    """Execute a BigQuery SQL query and return results as a DataFrame."""
    if client is None:
        client = get_client()
    return client.query(query).to_dataframe()


def fetch_weekly_tag_counts(client: bigquery.Client | None = None) -> pd.DataFrame:
    """Fetch weekly question counts for the top 10 SO tags since 2009."""
    df = run_query(TOP_TAGS_QUERY, client)
    df["week"] = pd.to_datetime(df["week"])
    return df


if __name__ == "__main__":
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_DATA_DIR / "tag_question_counts.csv"

    print("Connecting to BigQuery...")
    client = get_client()

    print("Running query for weekly tag counts (top 10 tags, 2009–present)...")
    df = fetch_weekly_tag_counts(client)

    print(f"Retrieved {len(df)} rows")
    print(f"Tags: {sorted(df['tag'].unique())}")
    print(f"Date range: {df['week'].min()} to {df['week'].max()}")

    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
