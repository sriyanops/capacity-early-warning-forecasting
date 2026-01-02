# Capacity Early Warning & Forecasting (CEW)

A Python-based forecasting and capacity decision-support tool for identifying short-horizon demand risk and capacity overload.  
The system generates demand forecasts, evaluates site-level capacity utilization, applies transparent risk tiers, and produces prioritized mitigation actions with documented model accuracy.

This project is designed for operations analysts, planners, and reviewers, and reflects how early-warning and planning tools are typically structured in enterprise environments — with an emphasis on interpretability, auditability, and actionable outputs rather than black-box optimization.

---

## What This Tool Does

### Problem This Tool Solves

Capacity constraints are frequently identified only after demand has exceeded available resources, resulting in service delays and reactive mitigation. Forecasts and utilization metrics are often reviewed separately, limiting their usefulness for proactive planning. This tool integrates demand forecasting with capacity thresholds to provide early warning signals and actionable guidance before constraints become critical. The system ingests daily, site-level volume data and produces:

- Short-horizon demand forecasts (baseline, interpretable)
- Capacity utilization estimates by site and day
- Rule-based utilization risk tiers:
  - **GREEN** — within capacity
  - **YELLOW** — approaching capacity
  - **RED** — capacity exceedance risk
- Prioritized mitigation actions tied to utilization risk
- Forecast accuracy backtests (MAE, MAPE) for confidence assessment

The outputs are designed to support:
- Capacity planning
- Near-term operational reviews
- Early warning and escalation
- Tactical resource allocation decisions

---

## Tech Stack

- **Python** — core language  
- **Pandas** — data manipulation and analytics  
- **Matplotlib** — forecast and trend visualization  
- **ReportLab** — programmatic executive PDF generation  

---

## Inputs

The tool operates on tabular, site-level volume data, where each row represents a site-day observation.

- [`daily_volume_sample.csv`](data/sample/daily_volume_sample.csv) — daily demand/volume by site  

Key inputs include:
- Site identifier
- Observation date
- Daily volume / demand

> **DISCLAIMER:**  
> The included dataset is synthetic and created solely to demonstrate system logic and structure.  
> No proprietary or sensitive operational data is used.

---

## Key Outputs

### 1. Executive PDF Report
Generated with a single command at runtime and includes:
- Executive summary of top capacity risks
- Overall demand forecast visualization
- Capacity utilization risk tables
- Prioritized recommended actions
- Forecast backtest accuracy (MAE / MAPE)
- Detailed appendix for analyst review
> Note: The PDF report is generated dynamically and is not committed to the repository.

### 2. Structured CSV Outputs
Generated at runtime for downstream analysis:
- Forecasts by site and day  
- Top-risk site listings  
- Decision recommendations  
- Backtest accuracy summaries (overall and by site)
> Note: CSV outputs are produced during execution and are intentionally excluded from version control.

---

## Project Structure

```text
capacity_early_warning/
├── data/
│   └── sample/
│       └── daily_volume_sample.csv
├── src/
│   ├── load.py            # Data ingestion
│   ├── forecast.py        # Forecasting logic
│   ├── capacity.py        # Capacity & utilization calculations
│   ├── decision_rules.py  # Risk tiers & mitigation actions
│   ├── backtest.py        # Forecast accuracy evaluation
│   ├── report_pdf.py      # Executive PDF generation
│   └── main.py            # Orchestration entry point
├── requirements.txt
└── README.md
```
> Note: Output files (CSVs, reports, PDFs) are generated at runtime and are intentionally not committed to the repository.
## How to run
> All commands must be run from repository root.

### Install dependencies 
```bash
pip install -r requirements.txt
```
### Generate forecasts, decisions, and executive report
```bash
python src/main.py
```
## Design Choices
### Interpretable Forecasting

A seasonal-naive baseline was chosen to prioritize transparency and explainability.
This mirrors real-world early-warning systems where trust and clarity often outweigh marginal accuracy gains.

### Rule-Based Capacity Tiers

Capacity risk is classified using explicit, deterministic rules rather than opaque scoring models.
This ensures decisions are traceable and defensible.

### Backtest-Driven Confidence

Forecast accuracy is explicitly reported (MAE, MAPE) so users understand when forecasts should be treated as:

actionable signals, or

directional early warnings.

### Separation of Concerns

Forecasting, capacity evaluation, decision logic, and reporting are decoupled.
This allows future extensions without rewriting the core pipeline.

### Limitations

Forecasts are short-horizon and intended for early warning, not long-term planning.

Capacity is assumed fixed per site for demonstration purposes.

External drivers (e.g., labor, weather, supplier variability) are not explicitly modeled.

## Future Enhancements

Alternative forecasting models

Scenario-based capacity simulation

Dynamic capacity assumptions

Automated alerting and escalation logic

Integration with live operational data

## Summary

This project demonstrates how demand forecasts can be translated into:

Clear capacity risk signals

Prioritized mitigation actions

Executive- and analyst-ready decision artifacts

It reflects real-world practices used in operations planning, logistics, and capacity management teams.

