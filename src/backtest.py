from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    by_site: pd.DataFrame         # site, n, mae, mape
    overall: pd.DataFrame         # n, mae, mape


def backtest_seasonal_naive(
    df: pd.DataFrame,
    horizon_days: int = 7,
    lookback_days: int = 56,
    date_col: str = "date",
    site_col: str = "site",
    y_col: str = "volume",
) -> BacktestResult:
    """
    Rolling backtest for a seasonal-naive baseline (t-7).

    Approach (simple + honest):
      - For each site, create naive predictions yhat = y(t-7)
      - Evaluate only on last `lookback_days` where t-7 exists
      - Report MAE and MAPE
    """
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d = d.sort_values([site_col, date_col]).reset_index(drop=True)

    # seasonal naive fitted values
    d["yhat"] = d.groupby(site_col)[y_col].shift(7)

    # restrict to the most recent lookback window with valid yhat
    max_date = d[date_col].max()
    window_start = max_date - pd.Timedelta(days=lookback_days)
    eval_df = d[(d[date_col] >= window_start) & (d["yhat"].notna())].copy()

    if eval_df.empty:
        raise ValueError("Backtest window produced no rows. Increase lookback_days or ensure >= 14 days of data/site.")

    eval_df["err"] = (eval_df[y_col] - eval_df["yhat"]).abs()

    # MAPE: avoid divide-by-zero by excluding y=0 rows
    eval_df["ape"] = np.where(
        eval_df[y_col].astype(float) == 0.0,
        np.nan,
        eval_df["err"] / eval_df[y_col].astype(float),
    )

    by_site = (
        eval_df.groupby(site_col, as_index=False)
        .agg(
            n=(y_col, "count"),
            mae=("err", "mean"),
            mape=("ape", "mean"),
        )
    )

    overall = pd.DataFrame(
        {
            "n": [int(eval_df[y_col].count())],
            "mae": [float(eval_df["err"].mean())],
            "mape": [float(eval_df["ape"].mean())],
        }
    )

    # clean formatting for downstream report
    by_site["mae"] = by_site["mae"].astype(float)
    by_site["mape"] = by_site["mape"].astype(float)
    overall["mae"] = overall["mae"].astype(float)
    overall["mape"] = overall["mape"].astype(float)

    return BacktestResult(by_site=by_site, overall=overall)
