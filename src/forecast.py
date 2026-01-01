from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForecastResult:
    horizon_days: int
    by_site_day: pd.DataFrame  # columns: date, site, yhat, yhat_lower, yhat_upper
    overall_day: pd.DataFrame  # columns: date, yhat, yhat_lower, yhat_upper


def _rolling_sigma(residuals: pd.Series, window: int = 28) -> float:
    """
    Robust-ish residual scale estimate for intervals.
    Falls back to global std if not enough data.
    """
    r = residuals.dropna()
    if len(r) < 10:
        return float(r.std(ddof=1)) if len(r) >= 2 else 0.0
    return float(r.tail(window).std(ddof=1))


def baseline_forecast(
    df: pd.DataFrame,
    horizon_days: int = 14,
    volume_col: str = "volume",
    date_col: str = "date",
    site_col: str = "site",
) -> ForecastResult:
    """
    Seasonal-naive (t-7) baseline forecast with rolling-mean fallback.
    Produces simple prediction intervals using recent residual volatility.
    """
    # Ensure expected ordering
    df = df.sort_values([site_col, date_col]).reset_index(drop=True)

    sites = sorted(df[site_col].unique())
    last_date = pd.to_datetime(df[date_col]).max().date()
    future_dates = [last_date + timedelta(days=i) for i in range(1, horizon_days + 1)]

    # Build lookup for fast access: (site, date) -> volume
    vol_lookup: Dict[Tuple[str, pd.Timestamp], float] = {}
    for row in df[[site_col, date_col, volume_col]].itertuples(index=False):
        vol_lookup[(getattr(row, site_col), pd.Timestamp(getattr(row, date_col)))] = float(
            getattr(row, volume_col)
        )

    # Precompute per-site rolling mean (last 7 days)
    by_site = {s: g.copy() for s, g in df.groupby(site_col)}
    roll_mean_7: Dict[str, float] = {}
    for s, g in by_site.items():
        tail = g.tail(7)[volume_col].astype(float)
        roll_mean_7[s] = float(tail.mean()) if len(tail) else 0.0

    # Estimate residual volatility per site based on in-sample one-week-ahead naive errors
    sigma_by_site: Dict[str, float] = {}
    for s, g in by_site.items():
        g = g.sort_values(date_col)
        # naive fitted = volume(t-7)
        fitted = g[volume_col].shift(7).astype(float)
        resid = g[volume_col].astype(float) - fitted
        sigma_by_site[s] = _rolling_sigma(resid, window=28)

    rows = []
    z = 1.96  # ~95% interval

    for s in sites:
        sigma = sigma_by_site.get(s, 0.0)
        fallback = roll_mean_7.get(s, 0.0)

        for d in future_dates:
            d_ts = pd.Timestamp(d)

            # seasonal naive: same weekday last week
            ref_ts = d_ts - pd.Timedelta(days=7)
            yhat = vol_lookup.get((s, ref_ts), None)

            if yhat is None:
                yhat = fallback

            yhat = float(max(0.0, yhat))
            lower = max(0.0, yhat - z * sigma)
            upper = max(0.0, yhat + z * sigma)

            rows.append(
                {
                    "date": d_ts,
                    "site": s,
                    "yhat": yhat,
                    "yhat_lower": lower,
                    "yhat_upper": upper,
                }
            )

    by_site_day = pd.DataFrame(rows).sort_values(["site", "date"]).reset_index(drop=True)

    overall_day = (
        by_site_day.groupby("date", as_index=False)[["yhat", "yhat_lower", "yhat_upper"]]
        .sum()
        .sort_values("date")
        .reset_index(drop=True)
    )

    return ForecastResult(horizon_days=horizon_days, by_site_day=by_site_day, overall_day=overall_day)
