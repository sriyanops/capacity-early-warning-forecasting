from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set

import pandas as pd
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image,
)


@dataclass(frozen=True)
class ReportInputs:
    overall_day: pd.DataFrame
    decisions: pd.DataFrame
    top_risk: pd.DataFrame
    backtest_overall: pd.DataFrame
    backtest_by_site: pd.DataFrame


def _save_forecast_chart(overall_day: pd.DataFrame, out_path: Path) -> Path:
    """
    Save a simple forecast chart (overall yhat with interval) to a PNG.
    """
    df = overall_day.copy().sort_values("date")
    df["date"] = pd.to_datetime(df["date"])

    fig = plt.figure(figsize=(8.0, 3.2))
    ax = plt.gca()

    ax.plot(df["date"], df["yhat"], label="Forecast (yhat)")
    ax.fill_between(
        df["date"],
        df["yhat_lower"],
        df["yhat_upper"],
        alpha=0.2,
        label="~95% interval",
    )

    ax.set_title("Overall Forecast (Next Horizon)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def _make_table(
    df: pd.DataFrame,
    max_rows: int,
    col_order: list[str],
    col_labels: Optional[list[str]] = None,
    page_width: float = 6.8 * inch,
    wrap_cols: Optional[Set[str]] = None,
) -> Table:
    """
    Build a ReportLab Table that:
      - stays within page_width
      - wraps selected columns (e.g., action, rationale) using Paragraph
      - uses repeatRows=1 so header repeats on page breaks
    """
    wrap_cols = wrap_cols or set()

    view = df.copy()
    if col_order:
        view = view[col_order]
    view = view.head(max_rows)

    # Format to strings (then optionally convert wrapped cols to Paragraphs)
    for c in view.columns:
        if c == "utilization":
            view[c] = view[c].astype(float).map(lambda x: f"{x:.2f}")
        elif c in {"yhat", "capacity"}:
            view[c] = view[c].astype(float).map(lambda x: f"{x:.0f}")
        elif c in {"yhat_lower", "yhat_upper"}:
            view[c] = view[c].astype(float).map(lambda x: f"{x:.0f}")
        else:
            view[c] = view[c].astype(str)

    header = col_labels if col_labels else list(view.columns)

    wrap_style = ParagraphStyle(
        "wrap",
        fontName="Helvetica",
        fontSize=8.5,
        leading=10.5,
        alignment=TA_LEFT,
        wordWrap="CJK",  # aggressive wrap, handles long tokens better
    )

    data_rows = []
    for row in view.itertuples(index=False):
        row_vals = list(row)
        out_row = []
        for idx, col in enumerate(view.columns):
            val = row_vals[idx]
            if col in wrap_cols:
                out_row.append(Paragraph(val.replace("\n", "<br/>"), wrap_style))
            else:
                out_row.append(val)
        data_rows.append(out_row)

    data = [header] + data_rows

    # Column width heuristics (tuned for letter + your margins)
    width_map = {
        "date": 1.0 * inch,
        "site": 0.55 * inch,
        "tier": 0.55 * inch,
        "utilization": 0.60 * inch,
        "yhat": 0.75 * inch,
        "capacity": 0.75 * inch,
        "priority": 0.35 * inch,
        "action": 2.3 * inch,
        "rationale": 2.3 * inch,
        "yhat_lower": 0.65 * inch,
        "yhat_upper": 0.65 * inch,
    }

    # Build colWidths list
    col_widths: list[float | None] = []
    for col in view.columns:
        col_widths.append(width_map.get(col, None))

    fixed = sum(w for w in col_widths if w is not None)  # type: ignore[arg-type]
    missing_idx = [i for i, w in enumerate(col_widths) if w is None]
    remaining = max(0.0, page_width - fixed)

    if missing_idx:
        each = remaining / len(missing_idx)
        for i in missing_idx:
            col_widths[i] = each

    total = sum(col_widths)  # type: ignore[arg-type]
    if total > page_width and total > 0:
        scale = page_width / total
        col_widths = [float(w) * scale for w in col_widths]  # type: ignore[arg-type]

    t = Table(data, colWidths=col_widths, hAlign="LEFT", repeatRows=1)

    style = TableStyle(
        [
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9EEF6")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#C9D2E3")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FC")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]
    )

    # Tier emphasis (use original string values from `view`)
    if "tier" in view.columns:
        tier_idx = list(view.columns).index("tier")
        for r in range(1, len(data)):
            tier_val = view.iloc[r - 1, tier_idx]
            if tier_val == "RED":
                style.add("TEXTCOLOR", (tier_idx, r), (tier_idx, r), colors.HexColor("#B00020"))
                style.add("FONTNAME", (tier_idx, r), (tier_idx, r), "Helvetica-Bold")
            elif tier_val == "YELLOW":
                style.add("TEXTCOLOR", (tier_idx, r), (tier_idx, r), colors.HexColor("#8A6D00"))
                style.add("FONTNAME", (tier_idx, r), (tier_idx, r), "Helvetica-Bold")

    t.setStyle(style)
    return t


def build_pdf_report(
    out_pdf_path: str | Path,
    inputs: ReportInputs,
    title: str = "Capacity Early Warning & Forecasting Report",
    subtitle: str = "Forecast-driven utilization risk tiers and mitigation recommendations",
) -> Path:
    out_pdf_path = Path(out_pdf_path)
    out_pdf_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    h1 = styles["Heading1"]
    h2 = styles["Heading2"]

    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        spaceAfter=8,
    )
    small = ParagraphStyle(
        "Small",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#334155"),
    )

    doc = SimpleDocTemplate(
        str(out_pdf_path),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title=title,
    )

    story = []

    # Title
    story.append(Paragraph(title, h1))
    story.append(Paragraph(subtitle, small))
    story.append(Spacer(1, 0.2 * inch))

    # Executive Summary: Top Risks
    story.append(Paragraph("Executive Summary (Top Risks)", h2))
    top_risk = inputs.top_risk.copy()
    if "date" in top_risk.columns:
        top_risk["date"] = pd.to_datetime(top_risk["date"]).dt.date.astype(str)

    top_risk_cols = ["date", "site", "yhat", "capacity", "utilization", "tier"]
    story.append(
        _make_table(
            top_risk,
            max_rows=12,
            col_order=top_risk_cols,
            col_labels=["Date", "Site", "Forecast", "Capacity", "Util", "Tier"],
        )
    )
    story.append(Spacer(1, 0.15 * inch))

    # Forecast chart
    story.append(Paragraph("Overall Forecast", h2))
    chart_path = out_pdf_path.parent / "forecast_overall.png"
    _save_forecast_chart(inputs.overall_day, chart_path)
    story.append(Image(str(chart_path), width=6.8 * inch, height=2.7 * inch))
    story.append(Spacer(1, 0.2 * inch))

    # --- Model accuracy section ---
    story.append(Paragraph("Model Accuracy (Backtest)", h2))

    bt_overall = inputs.backtest_overall.copy()
    bt_by_site = inputs.backtest_by_site.copy()

    # Format for display
    if "mae" in bt_overall.columns:
        bt_overall["mae"] = bt_overall["mae"].astype(float).map(lambda x: f"{x:.1f}")
    if "mape" in bt_overall.columns:
        bt_overall["mape"] = bt_overall["mape"].astype(float).map(lambda x: f"{x * 100:.1f}%")

    if "mae" in bt_by_site.columns:
        bt_by_site["mae"] = bt_by_site["mae"].astype(float).map(lambda x: f"{x:.1f}")
    if "mape" in bt_by_site.columns:
        bt_by_site["mape"] = bt_by_site["mape"].astype(float).map(lambda x: f"{x * 100:.1f}%")

    story.append(
        _make_table(
            bt_overall,
            max_rows=1,
            col_order=["n", "mae", "mape"],
            col_labels=["N (days)", "MAE (units)", "MAPE"],
        )
    )
    story.append(Spacer(1, 0.12 * inch))

    story.append(Paragraph("By Site (last 56-day window)", small))
    story.append(
        _make_table(
            bt_by_site.sort_values("mape", ascending=False),
            max_rows=10,
            col_order=["site", "n", "mae", "mape"],
            col_labels=["Site", "N (days)", "MAE (units)", "MAPE"],
        )
    )
    story.append(Spacer(1, 0.2 * inch))

    # Interpretation (NEW)
    story.append(
        Paragraph(
            "Interpretation: Lower MAPE is better; use this as a confidence check for short-horizon planning. "
            "Higher MAPE sites should be treated as directional early-warning signals (not precise capacity plans).",
            body,
        )
    )
    story.append(Spacer(1, 0.12 * inch))
    # --- end model accuracy section ---

    # Recommended actions — split into two tables for clean layout
    story.append(Paragraph("Recommended Actions (Highest Priority First)", h2))

    # Legend for P
    story.append(
        Paragraph(
            "<b>Priority (P):</b> 1 = Immediate action (RED), "
            "2 = Prep / monitor (YELLOW), "
            "4 = No action (GREEN).",
            small,
        )
    )
    story.append(Spacer(1, 0.08 * inch))

    decisions = inputs.decisions.copy()
    if "date" in decisions.columns:
        decisions["date"] = pd.to_datetime(decisions["date"]).dt.date.astype(str)

    metric_cols = ["date", "site", "tier", "utilization", "yhat", "capacity", "priority"]
    story.append(
        _make_table(
            decisions,
            max_rows=15,
            col_order=metric_cols,
            col_labels=["Date", "Site", "Tier", "Util", "Forecast", "Cap", "P"],
        )
    )
    story.append(Spacer(1, 0.12 * inch))

    action_cols = ["date", "site", "action"]
    story.append(
        _make_table(
            decisions,
            max_rows=15,
            col_order=action_cols,
            col_labels=["Date", "Site", "Action"],
            wrap_cols={"action"},
        )
    )

    story.append(Spacer(1, 0.15 * inch))
    story.append(
        Paragraph(
            "Notes: Forecasts use a seasonal-naive baseline (t-7) with rolling-mean fallback. "
            "Utilization tiers are rule-based for early warning—not exact capacity planning.",
            body,
        )
    )

    # Appendix
    story.append(PageBreak())
    story.append(Paragraph("Appendix: Forecast Detail (Sample)", h2))

    detail = inputs.decisions.copy()
    if "date" in detail.columns:
        detail["date"] = pd.to_datetime(detail["date"]).dt.date.astype(str)

    detail_cols = ["date", "site", "yhat", "yhat_lower", "yhat_upper", "capacity", "utilization", "tier", "priority"]
    story.append(
        _make_table(
            detail,
            max_rows=25,
            col_order=detail_cols,
            col_labels=["Date", "Site", "Yhat", "Lo", "Hi", "Cap", "Util", "Tier", "P"],
        )
    )

    doc.build(story)
    return out_pdf_path
