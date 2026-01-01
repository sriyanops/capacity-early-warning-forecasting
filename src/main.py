from __future__ import annotations

from pathlib import Path

from load import load_daily_volume
from forecast import baseline_forecast
from capacity import compute_capacity_status
from decision_rules import apply_decision_rules
from report_pdf import ReportInputs, build_pdf_report
from backtest import backtest_seasonal_naive


def main() -> None:
    root = Path(__file__).resolve().parents[1]

    data_path = root / "data" / "sample" / "daily_volume_sample.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data file: {data_path}")

    out_dir = root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load
    df = load_daily_volume(data_path)

    # 1b) Backtest (56-day window)
    bt = backtest_seasonal_naive(df, lookback_days=56)
    bt.by_site.to_csv(out_dir / "backtest_by_site_56d.csv", index=False)
    bt.overall.to_csv(out_dir / "backtest_overall_56d.csv", index=False)

    # 2) Forecast
    fc = baseline_forecast(df, horizon_days=14)

    # 3) Capacity utilization + tiers
    cap_status = compute_capacity_status(fc.by_site_day, df)

    # 4) Decision rules
    decisions = apply_decision_rules(cap_status.by_site_day)

    # 5) PDF report
    pdf_path = out_dir / "capacity_early_warning_report.pdf"
    build_pdf_report(
        pdf_path,
        ReportInputs(
            overall_day=fc.overall_day,
            decisions=decisions,
            top_risk=cap_status.top_risk,
            backtest_overall=bt.overall,
            backtest_by_site=bt.by_site,
        ),
    )
    print("PDF written to:", pdf_path)

    # 6) Save outputs
    fc.by_site_day.to_csv(out_dir / "forecast_by_site_day.csv", index=False)
    fc.overall_day.to_csv(out_dir / "forecast_overall_day.csv", index=False)
    cap_status.top_risk.to_csv(out_dir / "top_risk_next_14_days.csv", index=False)
    decisions.to_csv(out_dir / "decision_recommendations.csv", index=False)

    # Console summary
    print("\n=== CAPACITY EARLY WARNING: TOP RISKS (NEXT 14 DAYS) ===")
    print(
        cap_status.top_risk[
            ["date", "site", "yhat", "capacity", "utilization", "tier"]
        ]
        .head(15)
        .to_string(index=False)
    )

    print("\n=== TOP DECISIONS (HIGHEST PRIORITY FIRST) ===")
    print(
        decisions[
            ["date", "site", "yhat", "capacity", "utilization", "tier", "priority", "action"]
        ]
        .head(15)
        .to_string(index=False)
    )

    print("\nOutputs written to:", out_dir)


if __name__ == "__main__":
    main()
