"""PDF report for one reporting date.

Hackathon rule: you may change everything INSIDE build_report, but NOT its
name, its signature (result, path), or where it writes (path).
"""

from __future__ import annotations

import html
import io
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,  # noqa: E402
                                TableStyle)

from src.config import STRATEGIC_BENCHMARK  # noqa: E402
from src.utils import data, pit  # noqa: E402

STYLES = getSampleStyleSheet()


# LLMs love typographic unicode (non-breaking hyphens, thin spaces...) that the
# built-in PDF fonts cannot render (shows as black boxes). Map to plain ASCII.
_UNICODE_FIX = str.maketrans({"\u2011": "-", "\u2010": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
                              "\u2212": "-", "\u202f": " ", "\u2009": " ", "\u00a0": " ", "\u200b": "",
                              "\u2248": "~", "\u2264": "<=", "\u2265": ">=", "\u2192": "->",
                              "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u0394": "chg "})


def _md(text: str) -> list:
    """Very small markdown -> reportlab paragraphs (bold, headings, bullets)."""
    text = (text or "").translate(_UNICODE_FIX)
    out = []
    parts = text.split("```")
    if len(parts) > 1:  # odd parts are fenced code blocks -> monospace
        for i, part in enumerate(parts):
            out += [_pre(part.strip("\n"))] if i % 2 else _md(part)
        return out
    table_rows: list[list[str]] = []
    for block in text.split("\n") + [""]:
        if block.strip().startswith("|"):  # markdown table row
            cells = [c.strip() for c in block.strip().strip("|").split("|")]
            if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                table_rows.append(cells)
            continue
        if table_rows:
            out.append(_md_table(table_rows))
            table_rows = []
        if re.fullmatch(r"\s*(-{3,}|\*{3,})\s*", block):  # horizontal rule
            continue
        line = html.escape(block.rstrip())
        if not line.strip():
            continue
        line = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)
        if line.lstrip().startswith("#"):
            out.append(Paragraph(line.lstrip("# "), STYLES["Heading4"]))
        elif line.lstrip().startswith(("- ", "* ", "• ")):
            out.append(Paragraph("&bull; " + line.lstrip()[2:], STYLES["BodyText"]))
        else:
            out.append(Paragraph(line, STYLES["BodyText"]))
    return out


def _md_table(rows: list[list[str]]) -> Table:
    n = max(len(r) for r in rows)
    cell = STYLES["BodyText"].clone("cell", fontSize=7, leading=8)
    data_ = [[Paragraph(re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html.escape(c)), cell) for c in r + [""] * (n - len(r))]
             for r in rows]
    t = Table(data_, hAlign="LEFT")
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                           ("GRID", (0, 0), (-1, -1), 0.25, colors.grey)]))
    return t


def _pre(text: str) -> Paragraph:
    return Paragraph("<font face='Courier' size='6'>" + html.escape(text).replace("\n", "<br/>")
                     .replace(" ", "&nbsp;") + "</font>", STYLES["BodyText"])


def _fig_to_image(fig, width_cm=17) -> Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = Image(buf)
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth, img.drawHeight = width_cm * cm, width_cm * cm * ratio
    return img


def _table(df: pd.DataFrame, fmt="{:.1%}") -> Table:
    rows = [[df.index.name or ""] + list(df.columns)]
    for idx, r in df.iterrows():
        rows.append([str(idx)] + [fmt.format(v) if isinstance(v, (int, float)) else str(v) for v in r])
    t = Table(rows, hAlign="LEFT")
    t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 7), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                           ("GRID", (0, 0), (-1, -1), 0.25, colors.grey)]))
    return t


def _chart_weights(w: pd.Series, b: pd.Series) -> Image:
    df = pd.DataFrame({"portfolio": w, "benchmark": b}).fillna(0)
    df = df[(df.portfolio > 0) | (df.benchmark > 0)].sort_values("portfolio")
    fig, ax = plt.subplots(figsize=(8, 5))
    df.plot.barh(ax=ax, width=0.8)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Portfolio vs strategic benchmark")
    return _fig_to_image(fig)


def _chart_markets(as_of: str) -> Image:
    lv = pit.assets_asof(as_of).iloc[-253:]
    meta = data.read_assets_metadata().set_index("ticker")
    fin = pit.financial_asof(as_of).iloc[-253:]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    rets = lv.pct_change(fill_method=None).fillna(0)
    for cls, g in meta.groupby("asset_class"):
        (1 + rets[g.index].mean(axis=1)).cumprod().plot(ax=axes[0], label=cls)
    axes[0].set_title("Asset classes, 12m (equal-wt, local ccy)")
    axes[0].legend(fontsize=7)
    fin[["DE_10Y_YLD", "US_10Y_YLD", "IT_10Y_YLD"]].plot(ax=axes[1])
    axes[1].set_title("10Y government yields (%)")
    axes[1].legend(fontsize=7)
    fin[["VIX", "VSTOXX"]].plot(ax=axes[2])
    axes[2].set_title("Implied volatility")
    axes[2].legend(fontsize=7)
    for ax in axes:
        ax.set_xlabel("")
    fig.tight_layout()
    return _fig_to_image(fig, 18)


def build_report(result, path):
    """Write the PDF report for one reporting date. DO NOT change name/signature."""
    as_of = result["as_of_date"]
    path = Path(path)
    w = pd.Series(result["weights"])
    b = pd.Series(STRATEGIC_BENCHMARK)
    m = result.get("risk_metrics", {})
    expo = result.get("exposures", {})

    story = [Paragraph(f"Market Intelligence &amp; Allocation Report — {as_of}", STYLES["Title"]),
             Paragraph("<i>Practice run on SYNTHETIC data.</i>", STYLES["BodyText"]),
             Spacer(1, 6),
             Paragraph("1. Market intelligence", STYLES["Heading2"]),
             _chart_markets(as_of)]
    story += _md(result.get("analyst_summary", ""))

    story += [PageBreak(), Paragraph("2. Positioning", STYLES["Heading2"]), _chart_weights(w, b)]
    story += _md(result.get("pm_rationale", ""))
    for field, title in [("asset_class", "Asset class"), ("jurisdiction", "Jurisdiction"), ("currency", "Currency")]:
        if field in expo:
            df = pd.DataFrame({"portfolio": pd.Series(expo[field])})
            df.index.name = title
            story += [Spacer(1, 4), _table(df)]

    sig = pd.DataFrame(result.get("signals", []))
    if not sig.empty:
        sig = sig.set_index("ticker")[["view", "confidence", "reason"]]
        sig["reason"] = sig["reason"].str.slice(0, 90)
        story += [Spacer(1, 6), Paragraph("Analyst views", STYLES["Heading3"]), _table(sig, "{}")]

    story += [PageBreak(), Paragraph("3. Risk", STYLES["Heading2"])]
    if m:
        keys = ["ann_vol", "benchmark_ann_vol", "tracking_error", "var95_1d", "es95_1d", "var95_1m",
                "max_drawdown", "ann_return_hist"]
        rm = pd.DataFrame({"value": {k: m[k] for k in keys if k in m}})
        rm.index.name = f"Metric (EUR, hist. sim. {m.get('window', '')})"
        story += [_table(rm, "{:.2%}"), Paragraph(f"Modified duration: {m.get('duration', 0):.2f}",
                                                    STYLES["BodyText"])]
        rc = pd.DataFrame({"risk contribution": pd.Series(m.get("risk_contrib_by_class", {}))})
        rc.index.name = "Asset class"
        story += [Spacer(1, 4), _table(rc)]
    story += [Spacer(1, 6), Paragraph("Risk manager assessment", STYLES["Heading3"])]
    story += _md(result.get("risk_commentary", ""))
    if result.get("adjustments"):
        story += [Paragraph("Risk review adjustments (audit log)", STYLES["Heading3"])]
        story += [Paragraph("&bull; " + html.escape(str(a)), STYLES["BodyText"]) for a in result["adjustments"]]

    SimpleDocTemplate(str(path), pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm,
                      topMargin=1.5 * cm, bottomMargin=1.5 * cm).build(story)
