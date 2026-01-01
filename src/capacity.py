from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class CapacityStatus:
    by_site_day: pd.DataFrame
    top_risk: pd.DataFrame


def compute_capacity_status(
    forecast_by_site_day: pd.DataFrame,
    historical_df: pd.DataFrame,
    green_max: float = 0.85,
    yellow_max: float = 1.00,
) -> CapacityStatus:
    """
    Inputs:
      - forecast_by_site_day: columns [date, site, yhat, yhat_lower, yhat_upper]
      - historical_df: must include [site, capacity]

    Outputs:
      - by_site_day: forecast rows + capacity + utilization + tier
      - top_risk: sorted subset for quick viewing
    """
    # One capacity value per site (if multiple exist, take the most common)
    cap = (
        historical_df[["site", "capacity"]]
        .dropna()
        .assign(capacity=lambda x: x["capacity"].astype(int))
        .groupby("site", as_index=False)["capacity"]
        .agg(lambda s: int(s.mode().iat[0]) if not s.mode().empty else int(s.iloc[-1]))
    )

    out = forecast_by_site_day.merge(cap, on="site", how="left")
    if out["capacity"].isna().any():
        missing_sites = out.loc[out["capacity"].isna(), "site"].unique().tolist()
        raise ValueError(f"Missing capacity for sites: {missing_sites}")

    out["capacity"] = out["capacity"].astype(int)
    out["utilization"] = out["yhat"] / out["capacity"]

    def tier(u: float) -> str:
        if u <= green_max:
            return "GREEN"
        if u <= yellow_max:
            return "YELLOW"
        return "RED"

    out["tier"] = out["utilization"].apply(tier)

    # Useful flags for later decision rules
    out["over_capacity"] = out["utilization"] > 1.0
    out["near_capacity"] = (out["utilization"] > green_max) & (out["utilization"] <= 1.0)

    out = out.sort_values(["tier", "utilization"], ascending=[True, False]).reset_index(drop=True)

    top_risk = (
        out.sort_values(["utilization", "yhat"], ascending=[False, False])
        .head(25)
        .reset_index(drop=True)
    )

    return CapacityStatus(by_site_day=out, top_risk=top_risk)
