from __future__ import annotations

from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = {
    "date": "datetime64[ns]",
    "site": "object",
    "volume": "int64",
    "capacity": "int64",
}


def load_daily_volume(csv_path: str | Path) -> pd.DataFrame:
    """
    Load and validate daily site-level volume data.

    Required columns:
      - date (YYYY-MM-DD)
      - site (string)
      - volume (int)
      - capacity (int)

    Returns a cleaned, sorted DataFrame.
    """
    df = pd.read_csv(csv_path)

    # ---- schema check ----
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ---- type enforcement ----
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df["site"] = df["site"].astype(str)
    df["volume"] = df["volume"].astype(int)
    df["capacity"] = df["capacity"].astype(int)

    # ---- sanity checks ----
    if (df["volume"] < 0).any():
        raise ValueError("Negative volume detected")

    if (df["capacity"] <= 0).any():
        raise ValueError("Non-positive capacity detected")

    # ---- sort & index ----
    df = df.sort_values(["site", "date"]).reset_index(drop=True)

    return df
