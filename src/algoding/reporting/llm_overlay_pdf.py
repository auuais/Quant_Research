from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


REPORT_BG = colors.HexColor("#0b1220")
PANEL_BG = colors.HexColor("#121a2b")
HEADER_BG = colors.HexColor("#1f3b73")
ACCENT = colors.HexColor("#4da3ff")
TEXT = colors.HexColor("#e5eefc")
MUTED = colors.HexColor("#9db4d6")
POSITIVE = colors.HexColor("#38d39f")
NEGATIVE = colors.HexColor("#ff6b6b")
GRID = colors.HexColor("#24314a")

ACCENT_HEX = "#4da3ff"
POSITIVE_HEX = "#38d39f"
NEGATIVE_HEX = "#ff6b6b"
MUTED_HEX = "#9db4d6"
GRID_HEX = "#24314a"
REPORT_BG_HEX = "#0b1220"
TEXT_HEX = "#e5eefc"


@dataclass(frozen=True)
class AssetComparison:
    symbol: str
    baseline_strategy: str
    baseline_avg_total_return: float
    baseline_avg_max_drawdown: float
    best_overlay_strategy: str
    best_overlay_avg_total_return: float
    best_overlay_avg_max_drawdown: float
    return_delta: float
    drawdown_delta: float


def build_llm_overlay_comparison_pdf(
    *,
    hourly_artifact: str,
    output_path: str | None = None,
) -> dict[str, object]:
    artifact_path = Path(hourly_artifact)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    timeframe = str(payload.get("timeframe", _infer_timeframe_from_rows(payload))).lower()
    regular_hours_only = bool(payload.get("regular_hours_only", False))
    timeframe_label = _timeframe_label(timeframe=timeframe, regular_hours_only=regular_hours_only)
    output = Path(output_path) if output_path else Path(
        f"output/pdf/deepseek_{timeframe}{'_rth' if regular_hours_only and timeframe == 'hour' else ''}_overlay_vs_baseline_dark.pdf"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    asset_dir = output.parent / f"{output.stem}_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)

    comparisons = _build_asset_comparisons(payload)
    overlay_rows = [row for row in payload["leaderboard"] if not str(row["strategy_name"]).endswith("_baseline")]
    baseline_rows = [row for row in payload["leaderboard"] if str(row["strategy_name"]).endswith("_baseline")]
    top_rows = payload["leaderboard"][:12]

    baseline_return_stats = _stats([item.baseline_avg_total_return for item in comparisons])
    overlay_return_stats = _stats([item.best_overlay_avg_total_return for item in comparisons])
    baseline_drawdown_stats = _stats([item.baseline_avg_max_drawdown for item in comparisons])
    overlay_drawdown_stats = _stats([item.best_overlay_avg_max_drawdown for item in comparisons])

    better_assets = [item for item in comparisons if item.return_delta > 0]
    worse_assets = [item for item in comparisons if item.return_delta < 0]
    top_improvers = sorted(comparisons, key=lambda item: item.return_delta, reverse=True)[:6]
    top_decliners = sorted(comparisons, key=lambda item: item.return_delta)[:6]

    returns_chart = asset_dir / "returns_vs_baseline_dark.png"
    drawdowns_chart = asset_dir / "drawdown_vs_baseline_dark.png"
    return_stats_chart = asset_dir / "return_stats_dark.png"
    drawdown_stats_chart = asset_dir / "drawdown_stats_dark.png"

    _build_grouped_asset_chart(
        output_path=returns_chart,
        comparisons=comparisons,
        baseline_attr="baseline_avg_total_return",
        overlay_attr="best_overlay_avg_total_return",
        title=f"{timeframe_label} Avg Total Return: Baseline vs Best Overlay",
        percent=True,
    )
    _build_grouped_asset_chart(
        output_path=drawdowns_chart,
        comparisons=comparisons,
        baseline_attr="baseline_avg_max_drawdown",
        overlay_attr="best_overlay_avg_max_drawdown",
        title=f"{timeframe_label} Avg Max Drawdown: Baseline vs Best Overlay",
        percent=True,
    )
    _build_stats_chart(
        output_path=return_stats_chart,
        baseline_stats=baseline_return_stats,
        overlay_stats=overlay_return_stats,
        title="Aggregate Return Statistics Across 16 Assets",
        percent=False,
    )
    _build_stats_chart(
        output_path=drawdown_stats_chart,
        baseline_stats=baseline_drawdown_stats,
        overlay_stats=overlay_drawdown_stats,
        title="Aggregate Drawdown Statistics Across 16 Assets",
        percent=False,
    )

    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
        title=f"DeepSeek {timeframe_label} Overlay vs Baseline",
        author="Codex",
    )
    styles = _build_styles()
    story = [
        Paragraph(f"DeepSeek {timeframe_label} Overlay vs Baseline", styles["title"]),
        Spacer(1, 10),
        Paragraph(
            (
                "This report compares the DeepSeek news-sentiment overlay against the momentum baseline "
                f"for the same 16-stock universe on {timeframe_label.lower()} bars. The overlay branch uses the "
                "same execution-aware replay assumptions as the baseline."
            ),
            styles["body"],
        ),
        Spacer(1, 8),
        Paragraph("Run Setup", styles["section"]),
        _bullet_paragraph("Model: U:\\Deepseek\\models\\DeepSeek-R1-Distill-Qwen-1.5B", styles),
        _bullet_paragraph("Universe: 16 liquid U.S. stocks", styles),
        _bullet_paragraph("Windows: 1y, 2y, 3y", styles),
        _bullet_paragraph(f"Timeframe: {timeframe_label.lower()}", styles),
        _bullet_paragraph("Risk: 5% fixed stop, 1.5% trailing stop", styles),
        _bullet_paragraph("Execution-aware replay: spread, market impact, stop slippage, FINRA TAF, partial fills", styles),
        Spacer(1, 8),
        Paragraph("Headline Findings", styles["section"]),
        _bullet_paragraph(
            f"Best {timeframe_label.lower()} strategy overall was still a baseline row: {top_rows[0]['symbol']} / {top_rows[0]['strategy_name']} "
            f"with average total return {_pct(top_rows[0]['avg_total_return'])}.",
            styles,
        ),
        _bullet_paragraph(
            f"Across 48 symbol-window pairs, entry filter beat baseline {payload['baseline_vs_overlay']['summary']['entry_filter_better']} times, "
            f"exit filter {payload['baseline_vs_overlay']['summary']['exit_filter_better']} times, and combo {payload['baseline_vs_overlay']['summary']['combo_better']} times.",
            styles,
        ),
        _bullet_paragraph(
            f"Selecting the best overlay per asset, the overlay beat the baseline on {len(better_assets)}/16 assets and lagged on {len(worse_assets)}/16.",
            styles,
        ),
        _bullet_paragraph(
            f"Aggregate mean average return improved from {_pct(baseline_return_stats['mean'])} to {_pct(overlay_return_stats['mean'])}, "
            f"while mean average max drawdown improved from {_pct(baseline_drawdown_stats['mean'])} to {_pct(overlay_drawdown_stats['mean'])}.",
            styles,
        ),
        Spacer(1, 10),
        Paragraph(f"Top {timeframe_label} Rows", styles["section"]),
        _styled_table(
            [
                ["Rank", "Symbol", "Strategy", "Windows", "Avg Return", "Avg DD", "Score"],
                *[
                    [
                        str(index),
                        str(row["symbol"]),
                        str(row["strategy_name"]),
                        f"{row['profitable_windows']}/{len(row['windows_tested'])}",
                        _pct(float(row["avg_total_return"])),
                        _pct(float(row["avg_max_drawdown"])),
                        f"{float(row['composite_score']):.3f}",
                    ]
                    for index, row in enumerate(top_rows, start=1)
                ],
            ],
            col_widths=[0.35 * inch, 0.6 * inch, 2.5 * inch, 0.6 * inch, 0.8 * inch, 0.8 * inch, 0.6 * inch],
        ),
        Spacer(1, 12),
        Image(str(returns_chart), width=7.0 * inch, height=3.8 * inch),
        Spacer(1, 12),
        Image(str(drawdowns_chart), width=7.0 * inch, height=3.8 * inch),
        PageBreak(),
        Paragraph("Aggregate Statistics", styles["section"]),
        _styled_table(
            [
                ["Metric", "Baseline Return", "Best Overlay Return", "Baseline Drawdown", "Best Overlay Drawdown"],
                ["Mean", _pct(baseline_return_stats["mean"]), _pct(overlay_return_stats["mean"]), _pct(baseline_drawdown_stats["mean"]), _pct(overlay_drawdown_stats["mean"])],
                ["Variance", _raw(baseline_return_stats["variance"]), _raw(overlay_return_stats["variance"]), _raw(baseline_drawdown_stats["variance"]), _raw(overlay_drawdown_stats["variance"])],
                ["Std Dev", _pct(baseline_return_stats["stddev"]), _pct(overlay_return_stats["stddev"]), _pct(baseline_drawdown_stats["stddev"]), _pct(overlay_drawdown_stats["stddev"])],
                ["Min", _pct(baseline_return_stats["min"]), _pct(overlay_return_stats["min"]), _pct(baseline_drawdown_stats["min"]), _pct(overlay_drawdown_stats["min"])],
                ["Max", _pct(baseline_return_stats["max"]), _pct(overlay_return_stats["max"]), _pct(baseline_drawdown_stats["max"]), _pct(overlay_drawdown_stats["max"])],
            ],
            col_widths=[0.9 * inch, 1.2 * inch, 1.3 * inch, 1.3 * inch, 1.4 * inch],
        ),
        Spacer(1, 12),
        Image(str(return_stats_chart), width=7.0 * inch, height=3.3 * inch),
        Spacer(1, 12),
        Image(str(drawdown_stats_chart), width=7.0 * inch, height=3.3 * inch),
        PageBreak(),
        Paragraph("Best Overlay Per Asset", styles["section"]),
        _styled_table(
            [
                ["Symbol", "Baseline", "Best Overlay", "Base Return", "Overlay Return", "Delta", "Base DD", "Overlay DD"],
                *[
                    [
                        item.symbol,
                        _short_name(item.baseline_strategy),
                        _short_name(item.best_overlay_strategy),
                        _pct(item.baseline_avg_total_return),
                        _pct(item.best_overlay_avg_total_return),
                        _signed_pct(item.return_delta),
                        _pct(item.baseline_avg_max_drawdown),
                        _pct(item.best_overlay_avg_max_drawdown),
                    ]
                    for item in comparisons
                ],
            ],
            col_widths=[0.55 * inch, 1.2 * inch, 1.35 * inch, 0.8 * inch, 0.8 * inch, 0.75 * inch, 0.75 * inch, 0.8 * inch],
            body_font_size=7.2,
        ),
        Spacer(1, 10),
        Paragraph("Largest Positive Deltas", styles["section"]),
        _styled_table(
            [
                ["Symbol", "Overlay", "Return Delta", "Drawdown Delta"],
                *[
                    [item.symbol, _short_name(item.best_overlay_strategy), _signed_pct(item.return_delta), _signed_pct(item.drawdown_delta)]
                    for item in top_improvers
                ],
            ],
            col_widths=[0.7 * inch, 1.7 * inch, 1.1 * inch, 1.1 * inch],
        ),
        Spacer(1, 10),
        Paragraph("Largest Negative Deltas", styles["section"]),
        _styled_table(
            [
                ["Symbol", "Overlay", "Return Delta", "Drawdown Delta"],
                *[
                    [item.symbol, _short_name(item.best_overlay_strategy), _signed_pct(item.return_delta), _signed_pct(item.drawdown_delta)]
                    for item in top_decliners
                ],
            ],
            col_widths=[0.7 * inch, 1.7 * inch, 1.1 * inch, 1.1 * inch],
        ),
    ]

    doc.build(story, onFirstPage=_draw_dark_page, onLaterPages=_draw_dark_page)

    summary = {
        "output_path": str(output.resolve()),
        "artifact_path": str(artifact_path.resolve()),
        "timeframe": timeframe,
        "regular_hours_only": regular_hours_only,
        "baseline_vs_overlay_summary": payload["baseline_vs_overlay"]["summary"],
        "better_assets": len(better_assets),
        "worse_assets": len(worse_assets),
        "baseline_return_stats": baseline_return_stats,
        "overlay_return_stats": overlay_return_stats,
        "baseline_drawdown_stats": baseline_drawdown_stats,
        "overlay_drawdown_stats": overlay_drawdown_stats,
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _build_asset_comparisons(payload: dict[str, object]) -> list[AssetComparison]:
    by_symbol: dict[str, list[dict[str, object]]] = {}
    for row in payload["leaderboard"]:
        by_symbol.setdefault(str(row["symbol"]), []).append(row)

    comparisons: list[AssetComparison] = []
    for symbol, items in sorted(by_symbol.items()):
        baseline = next(item for item in items if str(item["strategy_name"]).endswith("_baseline"))
        overlays = [item for item in items if not str(item["strategy_name"]).endswith("_baseline")]
        best_overlay = max(overlays, key=lambda item: float(item["avg_total_return"]))
        comparisons.append(
            AssetComparison(
                symbol=symbol,
                baseline_strategy=str(baseline["strategy_name"]),
                baseline_avg_total_return=float(baseline["avg_total_return"]),
                baseline_avg_max_drawdown=float(baseline["avg_max_drawdown"]),
                best_overlay_strategy=str(best_overlay["strategy_name"]),
                best_overlay_avg_total_return=float(best_overlay["avg_total_return"]),
                best_overlay_avg_max_drawdown=float(best_overlay["avg_max_drawdown"]),
                return_delta=float(best_overlay["avg_total_return"]) - float(baseline["avg_total_return"]),
                drawdown_delta=float(best_overlay["avg_max_drawdown"]) - float(baseline["avg_max_drawdown"]),
            )
        )
    return comparisons


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(statistics.mean(values), 6),
        "variance": round(statistics.variance(values), 6) if len(values) > 1 else 0.0,
        "stddev": round(statistics.stdev(values), 6) if len(values) > 1 else 0.0,
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def _build_grouped_asset_chart(
    *,
    output_path: Path,
    comparisons: list[AssetComparison],
    baseline_attr: str,
    overlay_attr: str,
    title: str,
    percent: bool,
) -> None:
    labels = [item.symbol for item in comparisons]
    baseline_values = [getattr(item, baseline_attr) for item in comparisons]
    overlay_values = [getattr(item, overlay_attr) for item in comparisons]

    fig, ax = plt.subplots(figsize=(11.5, 6.2), facecolor=REPORT_BG_HEX)
    ax.set_facecolor(REPORT_BG_HEX)
    positions = list(range(len(labels)))
    width = 0.38
    ax.bar([value - width / 2 for value in positions], baseline_values, width=width, label="Baseline", color=ACCENT_HEX)
    ax.bar([value + width / 2 for value in positions], overlay_values, width=width, label="Best Overlay", color=POSITIVE_HEX)
    ax.set_title(title, color=TEXT_HEX, fontsize=14, pad=14)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", color=TEXT_HEX)
    ax.tick_params(axis="y", colors=TEXT_HEX)
    ax.grid(axis="y", color=GRID_HEX, alpha=0.55)
    for spine in ax.spines.values():
        spine.set_color(GRID_HEX)
    ax.legend(facecolor=REPORT_BG_HEX, edgecolor=GRID_HEX, labelcolor=TEXT_HEX)
    if percent:
        ax.yaxis.set_major_formatter(lambda value, _: f"{value * 100:.0f}%")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, facecolor=REPORT_BG_HEX)
    plt.close(fig)


def _build_stats_chart(
    *,
    output_path: Path,
    baseline_stats: dict[str, float],
    overlay_stats: dict[str, float],
    title: str,
    percent: bool,
) -> None:
    labels = ["mean", "variance", "stddev", "min", "max"]
    baseline_values = [baseline_stats[label] for label in labels]
    overlay_values = [overlay_stats[label] for label in labels]

    fig, ax = plt.subplots(figsize=(10.8, 5.4), facecolor=REPORT_BG_HEX)
    ax.set_facecolor(REPORT_BG_HEX)
    positions = list(range(len(labels)))
    width = 0.38
    ax.bar([value - width / 2 for value in positions], baseline_values, width=width, label="Baseline", color=ACCENT_HEX)
    ax.bar([value + width / 2 for value in positions], overlay_values, width=width, label="Best Overlay", color=POSITIVE_HEX)
    ax.set_title(title, color=TEXT_HEX, fontsize=14, pad=14)
    ax.set_xticks(positions)
    ax.set_xticklabels([label.title() for label in labels], color=TEXT_HEX)
    ax.tick_params(axis="y", colors=TEXT_HEX)
    ax.grid(axis="y", color=GRID_HEX, alpha=0.55)
    for spine in ax.spines.values():
        spine.set_color(GRID_HEX)
    ax.legend(facecolor=REPORT_BG_HEX, edgecolor=GRID_HEX, labelcolor=TEXT_HEX)
    if percent:
        ax.yaxis.set_major_formatter(lambda value, _: f"{value * 100:.0f}%")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, facecolor=REPORT_BG_HEX)
    plt.close(fig)


def _build_styles() -> dict[str, ParagraphStyle]:
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=TEXT,
            alignment=TA_CENTER,
        ),
        "section": ParagraphStyle(
            "section",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=ACCENT,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.6,
            leading=13,
            textColor=TEXT,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.6,
            leading=13,
            textColor=TEXT,
            leftIndent=12,
            bulletIndent=0,
        ),
    }


def _styled_table(rows: list[list[str]], *, col_widths: list[float], body_font_size: float = 8.0) -> Table:
    table = Table(rows, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
                ("TEXTCOLOR", (0, 0), (-1, 0), TEXT),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), body_font_size),
                ("BACKGROUND", (0, 1), (-1, -1), PANEL_BG),
                ("TEXTCOLOR", (0, 1), (-1, -1), TEXT),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), body_font_size),
                ("GRID", (0, 0), (-1, -1), 0.4, GRID),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    return table


def _draw_dark_page(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFillColor(REPORT_BG)
    canvas.rect(0, 0, doc.pagesize[0], doc.pagesize[1], fill=1, stroke=0)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(doc.pagesize[0] - 36, 18, f"Page {doc.page}")
    canvas.restoreState()


def _bullet_paragraph(text: str, styles: dict[str, ParagraphStyle]) -> Paragraph:
    return Paragraph(f"&bull; {text}", styles["bullet"])


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _signed_pct(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value * 100:.2f}%"


def _raw(value: float) -> str:
    return f"{value:.6f}"


def _short_name(strategy_name: str) -> str:
    return strategy_name.replace("_momentum_", "_").replace("_llm_", "_")


def _infer_timeframe_from_rows(payload: dict[str, object]) -> str:
    rows = payload.get("leaderboard", [])
    if not rows:
        return "day"
    window_names = rows[0].get("windows_tested", [])
    if not window_names:
        return "day"
    first = str(window_names[0]).lower()
    if "_hour" in first:
        return "hour"
    return "day"


def _timeframe_label(*, timeframe: str, regular_hours_only: bool) -> str:
    if timeframe == "hour":
        return "Hourly RTH" if regular_hours_only else "Hourly"
    if timeframe == "day":
        return "Daily"
    return timeframe.title()
