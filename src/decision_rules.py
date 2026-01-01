from __future__ import annotations

import pandas as pd


def apply_decision_rules(capacity_by_site_day: pd.DataFrame) -> pd.DataFrame:
    """
    Input expects columns:
      - date, site, yhat, capacity, utilization, tier

    Output adds:
      - priority (1 highest)
      - action (string)
      - rationale (string)
    """
    df = capacity_by_site_day.copy()

    def decide(row) -> tuple[int, str, str]:
        site = row["site"]
        u = float(row["utilization"])
        yhat = float(row["yhat"])
        cap = int(row["capacity"])
        tier = row["tier"]

        if tier == "GREEN":
            return (4, "No action", f"Forecast {yhat:.0f} vs capacity {cap} (util {u:.2f}).")

        if tier == "YELLOW":
            return (
                2,
                "Prep mitigation: pre-stage labor/trailers; monitor inbound variability",
                f"Near capacity: forecast {yhat:.0f} vs {cap} (util {u:.2f}) at {site}.",
            )

        # RED
        overload = yhat - cap
        return (
            1,
            "Mitigate overload: add labor hours; pull flex linehaul; shift/reroute volume to nearby site",
            f"Over capacity by ~{overload:.0f} units: forecast {yhat:.0f} vs {cap} (util {u:.2f}) at {site}.",
        )

    decisions = df.apply(decide, axis=1, result_type="expand")
    decisions.columns = ["priority", "action", "rationale"]
    df = pd.concat([df, decisions], axis=1)

    # Helpful ordering for report
    df = df.sort_values(["priority", "utilization"], ascending=[True, False]).reset_index(drop=True)
    return df
