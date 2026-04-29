from __future__ import annotations

import json
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
PREVIOUS = "#8b9dc3"
PROMPT = "#38d39f"
META = "#ffb454"
GRID = "#24314a"
TEXT_HEX = "#e5eefc"
GRID_HEX = "#24314a"
REPORT_BG_HEX = "#0b1220"


@dataclass(frozen=True)
class StrictComparisonRow:
    symbol: str
    previous_return: float
    previous_drawdown: float
    strict_prompt_return: float
    strict_prompt_drawdown: float
    strict_meta_return: float
    strict_meta_drawdown: float
    strategy_name: str


def build_deepseek_strict_compare_pdf(
    *,
    comparison_artifact: str,
    output_path: str | None = None,
) -> dict[str, object]:
    artifact_path = Path(comparison_artifact)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    output = Path(output_path) if output_path else Path("output/pdf/deepseek_top5_strict_exact_vs_meta_dark.pdf")
    output.parent.mkdir(parents=True, exist_ok=True)
    asset_dir = output.parent / f"{output.stem}_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        StrictComparisonRow(
            symbol=str(item["symbol"]),
            previous_return=float(item["previous_window_average"]["avg_total_return"]),
            previous_drawdown=float(item["previous_window_average"]["avg_max_drawdown"]),
            strict_prompt_return=float(item["strict_exact_prompt_variant"]["total_return"]),
            strict_prompt_drawdown=float(item["strict_exact_prompt_variant"]["max_drawdown"]),
            strict_meta_return=float(item["strict_meta_model"]["total_return"]),
            strict_meta_drawdown=float(item["strict_meta_model"]["max_drawdown"]),
            strategy_name=str(item["previous_window_average"]["strategy_name"]),
        )
        for item in payload["comparison_rows"]
    ]

    returns_chart = asset_dir / "returns_comparison.png"
    drawdowns_chart = asset_dir / "drawdowns_comparison.png"
    strict_stats_chart = asset_dir / "strict_stats.png"
    _build_grouped_chart(
        output_path=returns_chart,
        rows=rows,
        kind="return",
        title="DeepSeek Top-5 Stocks: Previous Window Average vs Strict Exact vs Meta",
    )
    _build_grouped_chart(
        output_path=drawdowns_chart,
        rows=rows,
        kind="drawdown",
        title="DeepSeek Top-5 Stocks: Drawdown Comparison",
    )
    _build_strict_stats_chart(
        output_path=strict_stats_chart,
        prompt_values=[row.strict_prompt_return for row in rows],
        meta_values=[row.strict_meta_return for row in rows],
    )

    styles = _build_styles()
    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
        title="DeepSeek Strict Top-5 Comparison",
        author="Codex",
    )
    story = [
        Paragraph("DeepSeek Top-5 Strict Comparison", styles["title"]),
        Spacer(1, 10),
        Paragraph(
            (
                "This report separates three different result types so the earlier DeepSeek stock winners are not "
                "mixed with the later strict unseen-test branch. The earlier window averages are context only. "
                "The apples-to-apples comparison is strict exact prompt variant vs strict fine-tuned meta-model."
            ),
            styles["body"],
        ),
        Spacer(1, 10),
        Paragraph("Headline", styles["section"]),
        _bullet(styles, f"Earlier DeepSeek top-5 multi-window average mean return: {_pct(payload['aggregates']['previous_window_average']['mean_total_return'])}"),
        _bullet(styles, f"Strict exact-variant mean return: {_pct(payload['aggregates']['strict_exact_prompt_variant']['mean_total_return'])}"),
        _bullet(styles, f"Strict meta-model mean return: {_pct(payload['aggregates']['strict_meta_model']['mean_total_return'])}"),
        _bullet(styles, f"Strict prompt variant beat the strict meta-model on {payload['aggregates']['strict_prompt_beats_meta']}/5 symbols."),
        Spacer(1, 10),
        Paragraph("Comparison Table", styles["section"]),
        _styled_table(
            [
                ["Symbol", "Prior Winner", "Prev Avg Return", "Prev Avg DD", "Strict Prompt Return", "Strict Prompt DD", "Strict Meta Return", "Strict Meta DD"],
                *[
                    [
                        row.symbol,
                        _short_name(row.strategy_name),
                        _pct(row.previous_return),
                        _pct(row.previous_drawdown),
                        _pct(row.strict_prompt_return),
                        _pct(row.strict_prompt_drawdown),
                        _pct(row.strict_meta_return),
                        _pct(row.strict_meta_drawdown),
                    ]
                    for row in rows
                ],
            ],
            col_widths=[0.55 * inch, 1.7 * inch, 0.8 * inch, 0.8 * inch, 0.95 * inch, 0.85 * inch, 0.95 * inch, 0.85 * inch],
            body_font_size=7.2,
        ),
        Spacer(1, 12),
        Image(str(returns_chart), width=7.1 * inch, height=3.8 * inch),
        Spacer(1, 12),
        Image(str(drawdowns_chart), width=7.1 * inch, height=3.8 * inch),
        PageBreak(),
        Paragraph("Strict Aggregate Statistics", styles["section"]),
        _styled_table(
            [
                ["Metric", "Strict Prompt", "Strict Meta"],
                *[
                    [label.title().replace("_", " "), _pct(values["prompt"]), _pct(values["meta"])]
                    for label, values in _strict_stats(rows).items()
                ],
            ],
            col_widths=[1.4 * inch, 1.3 * inch, 1.3 * inch],
        ),
        Spacer(1, 12),
        Image(str(strict_stats_chart), width=7.0 * inch, height=3.5 * inch),
        Spacer(1, 12),
        Paragraph("Interpretation", styles["section"]),
        _bullet(styles, "The earlier DeepSeek headline winners remain valid for the broad 1y/2y/3y window-averaging study."),
        _bullet(styles, "On the strict unseen later test slice, the exact earlier prompt variants still beat the fine-tuned news+price meta-model on return."),
        _bullet(styles, "The meta-model remains more defensive, but its first pass is too selective and often stays flat."),
    ]
    doc.build(story, onFirstPage=_draw_dark_page, onLaterPages=_draw_dark_page)

    summary = {
        "output_path": str(output.resolve()),
        "comparison_artifact": str(artifact_path.resolve()),
        "rows": len(rows),
        "aggregates": payload["aggregates"],
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _strict_stats(rows: list[StrictComparisonRow]) -> dict[str, dict[str, float]]:
    prompt_values = [row.strict_prompt_return for row in rows]
    meta_values = [row.strict_meta_return for row in rows]
    return {
        "mean": {"prompt": statistics.mean(prompt_values), "meta": statistics.mean(meta_values)},
        "std_dev": {"prompt": statistics.stdev(prompt_values) if len(prompt_values) > 1 else 0.0, "meta": statistics.stdev(meta_values) if len(meta_values) > 1 else 0.0},
        "min": {"prompt": min(prompt_values), "meta": min(meta_values)},
        "max": {"prompt": max(prompt_values), "meta": max(meta_values)},
    }


def _build_grouped_chart(*, output_path: Path, rows: list[StrictComparisonRow], kind: str, title: str) -> None:
    labels = [row.symbol for row in rows]
    if kind == "return":
        previous_values = [row.previous_return for row in rows]
        prompt_values = [row.strict_prompt_return for row in rows]
        meta_values = [row.strict_meta_return for row in rows]
    else:
        previous_values = [row.previous_drawdown for row in rows]
        prompt_values = [row.strict_prompt_drawdown for row in rows]
        meta_values = [row.strict_meta_drawdown for row in rows]
    fig, ax = plt.subplots(figsize=(11.2, 6.0), facecolor=REPORT_BG_HEX)
    ax.set_facecolor(REPORT_BG_HEX)
    positions = list(range(len(labels)))
    width = 0.24
    ax.bar([value - width for value in positions], previous_values, width=width, label="Previous Window Avg", color=PREVIOUS)
    ax.bar(positions, prompt_values, width=width, label="Strict Exact Prompt", color=PROMPT)
    ax.bar([value + width for value in positions], meta_values, width=width, label="Strict Meta", color=META)
    ax.set_title(title, color=TEXT_HEX, fontsize=14, pad=14)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", color=TEXT_HEX)
    ax.tick_params(axis="y", colors=TEXT_HEX)
    ax.grid(axis="y", color=GRID_HEX, alpha=0.55)
    for spine in ax.spines.values():
        spine.set_color(GRID_HEX)
    ax.legend(facecolor=REPORT_BG_HEX, edgecolor=GRID_HEX, labelcolor=TEXT_HEX)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value * 100:.0f}%")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, facecolor=REPORT_BG_HEX)
    plt.close(fig)


def _build_strict_stats_chart(*, output_path: Path, prompt_values: list[float], meta_values: list[float]) -> None:
    labels = ["Mean", "Std Dev", "Min", "Max"]
    prompt_stats = [
        statistics.mean(prompt_values),
        statistics.stdev(prompt_values) if len(prompt_values) > 1 else 0.0,
        min(prompt_values),
        max(prompt_values),
    ]
    meta_stats = [
        statistics.mean(meta_values),
        statistics.stdev(meta_values) if len(meta_values) > 1 else 0.0,
        min(meta_values),
        max(meta_values),
    ]
    fig, ax = plt.subplots(figsize=(9.8, 5.2), facecolor=REPORT_BG_HEX)
    ax.set_facecolor(REPORT_BG_HEX)
    positions = list(range(len(labels)))
    width = 0.35
    ax.bar([value - width / 2 for value in positions], prompt_stats, width=width, label="Strict Exact Prompt", color=PROMPT)
    ax.bar([value + width / 2 for value in positions], meta_stats, width=width, label="Strict Meta", color=META)
    ax.set_title("Strict Test Return Statistics", color=TEXT_HEX, fontsize=14, pad=14)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, color=TEXT_HEX)
    ax.tick_params(axis="y", colors=TEXT_HEX)
    ax.grid(axis="y", color=GRID_HEX, alpha=0.55)
    for spine in ax.spines.values():
        spine.set_color(GRID_HEX)
    ax.legend(facecolor=REPORT_BG_HEX, edgecolor=GRID_HEX, labelcolor=TEXT_HEX)
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


def _bullet(styles: dict[str, ParagraphStyle], text: str) -> Paragraph:
    return Paragraph(f"&bull; {text}", styles["bullet"])


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _short_name(name: str) -> str:
    return name.replace("_momentum_", "_").replace("_llm_", "_")
