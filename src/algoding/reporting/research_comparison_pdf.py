from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

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


@dataclass(frozen=True)
class StrategyRecord:
    branch: str
    experiment: str
    market: str
    symbol: str
    strategy_name: str
    windows_tested: list[str]
    profitable_windows: int
    avg_total_return: float
    avg_annualized_return: float
    avg_max_drawdown: float
    avg_profit_factor: float
    report_score: float
    note: str = ""


def build_research_comparison_pdf(
    *,
    output_path: str | None = None,
    dl_no_rule_artifact: str | None = None,
    dl_with_rule_artifact: str | None = None,
    ml_artifacts: list[str] | None = None,
    rule_artifacts: list[str] | None = None,
) -> dict[str, object]:
    resolved_output = Path(output_path) if output_path else Path("output/pdf/research_strategy_comparison_dark.pdf")
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    asset_dir = resolved_output.parent / f"{resolved_output.stem}_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)

    default_rule_artifacts = [
        "reports/research/slv_momentum_daily_hourly_1p5_no_same_day_compare.json",
        "reports/research/hourly_no_same_day_reentry_contenders.json",
    ]
    default_ml_artifacts = [
        "reports/research/ml_commodities_hourly_gpu_run.json",
        "reports/research/ml_etfs_stocks_hourly_gpu_run.json",
    ]
    resolved_rule_artifacts = [Path(path) for path in (rule_artifacts or default_rule_artifacts)]
    resolved_ml_artifacts = [Path(path) for path in (ml_artifacts or default_ml_artifacts)]
    resolved_dl_with_rule = Path(
        dl_with_rule_artifact or "reports/research/dl_commodities_stocks_hourly_gpu_run.json"
    )
    resolved_dl_no_rule = Path(
        dl_no_rule_artifact or "reports/research/dl_commodities_stocks_hourly_gpu_no_reentry_off_7w_run.json"
    )

    rule_records = _load_rule_based_records(resolved_rule_artifacts)
    ml_records = _load_leaderboard_records(
        resolved_ml_artifacts,
        branch="ML",
        default_experiment="hourly cost-aware, no-same-day ON",
    )
    dl_with_rule_records = _load_leaderboard_records(
        [resolved_dl_with_rule],
        branch="DL",
        default_experiment="hourly cost-aware, no-same-day ON",
    )
    dl_no_rule_records = _load_leaderboard_records(
        [resolved_dl_no_rule],
        branch="DL",
        default_experiment="hourly cost-aware, no-same-day OFF",
    )
    all_records = rule_records + ml_records + dl_with_rule_records + dl_no_rule_records
    if not all_records:
        raise RuntimeError("No strategy records were available to build the comparison PDF.")

    top_overall = sorted(all_records, key=lambda item: item.report_score, reverse=True)[:12]
    branch_leaders = _branch_leaders(all_records)
    dl_rule_delta_rows = _build_dl_rule_delta_rows(dl_with_rule_records, dl_no_rule_records)

    total_return_chart = asset_dir / "top_total_return_dark.png"
    score_chart = asset_dir / "top_report_score_dark.png"
    dl_toggle_chart = asset_dir / "dl_rule_toggle_dark.png"

    _build_bar_chart(
        output_path=total_return_chart,
        records=top_overall[:10],
        value_attr="avg_total_return",
        title="Top Strategies by Average Total Return",
        color=ACCENT_HEX,
        percent=True,
    )
    _build_bar_chart(
        output_path=score_chart,
        records=top_overall[:10],
        value_attr="report_score",
        title="Top Strategies by Unified Report Score",
        color=POSITIVE_HEX,
        percent=False,
    )
    if dl_rule_delta_rows:
        _build_grouped_toggle_chart(dl_toggle_chart, dl_rule_delta_rows[:8])

    best_overall = top_overall[0]
    best_no_rule = sorted(dl_no_rule_records, key=lambda item: item.report_score, reverse=True)[0]
    avg_dl_with_rule = mean(record.report_score for record in dl_with_rule_records) if dl_with_rule_records else 0.0
    avg_dl_no_rule = mean(record.report_score for record in dl_no_rule_records) if dl_no_rule_records else 0.0
    dl_rule_direction = "improved" if avg_dl_no_rule > avg_dl_with_rule else "worsened"

    doc = SimpleDocTemplate(
        str(resolved_output),
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
        title="Algoding Strategy Comparison",
        author="Codex",
    )
    styles = _build_styles()
    story = [
        Paragraph("Algoding Research Comparison", styles["title"]),
        Spacer(1, 10),
        Paragraph(
            (
                "This report compares the strongest rule-based, ML, and DL strategy branches under the "
                "execution-aware research path. The latest DL rerun disables the no-same-day reentry rule "
                "and extends the window set to 1m, 3m, 6m, 9m, 1y, 2y, and 3y."
            ),
            styles["body"],
        ),
        Spacer(1, 8),
        Paragraph("Headline Findings", styles["section"]),
        _bullet_paragraph(
            (
                f"Best overall recorded strategy: {best_overall.symbol} / {best_overall.strategy_name} "
                f"({best_overall.branch}, {best_overall.experiment}), average total return "
                f"{_pct(best_overall.avg_total_return)}, drawdown {_pct(best_overall.avg_max_drawdown)}."
            ),
            styles,
        ),
        _bullet_paragraph(
            (
                f"Best DL strategy in the new no-rule rerun: {best_no_rule.symbol} / {best_no_rule.strategy_name}, "
                f"average total return {_pct(best_no_rule.avg_total_return)}, profitable windows "
                f"{best_no_rule.profitable_windows}/{len(best_no_rule.windows_tested)}."
            ),
            styles,
        ),
        _bullet_paragraph(
            (
                f"Across the matched DL set, disabling no-same-day reentry {dl_rule_direction} the average report "
                f"score from {avg_dl_with_rule:.3f} to {avg_dl_no_rule:.3f}."
            ),
            styles,
        ),
        _bullet_paragraph(
            "Shared execution-aware assumptions for the hourly branch: 5% fixed stop, 1.5% trailing stop, "
            "regular-hours-only bars, spread, market impact, stop slippage, regulatory fees, and partial-fill caps.",
            styles,
        ),
        Spacer(1, 10),
        Paragraph("Branch Leaders", styles["section"]),
        _styled_table(
            [
                ["Branch", "Experiment", "Symbol", "Strategy", "Windows", "Avg Return", "Avg DD", "Score"],
                *[
                    [
                        branch,
                        record.experiment,
                        record.symbol,
                        record.strategy_name,
                        str(len(record.windows_tested)),
                        _pct(record.avg_total_return),
                        _pct(record.avg_max_drawdown),
                        f"{record.report_score:.3f}",
                    ]
                    for branch, record in branch_leaders
                ],
            ],
            col_widths=[0.8 * inch, 1.55 * inch, 0.55 * inch, 1.65 * inch, 0.45 * inch, 0.8 * inch, 0.8 * inch, 0.6 * inch],
        ),
        Spacer(1, 10),
        Paragraph("Top Strategies Overall", styles["section"]),
        _styled_table(
            [
                ["Rank", "Branch", "Symbol", "Strategy", "Experiment", "Win", "Avg Return", "Avg Ann.", "Avg DD", "Score"],
                *[
                    [
                        str(index),
                        record.branch,
                        record.symbol,
                        record.strategy_name,
                        record.experiment,
                        f"{record.profitable_windows}/{len(record.windows_tested)}",
                        _pct(record.avg_total_return),
                        _pct(record.avg_annualized_return),
                        _pct(record.avg_max_drawdown),
                        f"{record.report_score:.3f}",
                    ]
                    for index, record in enumerate(top_overall, start=1)
                ],
            ],
            col_widths=[0.35 * inch, 0.55 * inch, 0.5 * inch, 1.55 * inch, 1.6 * inch, 0.45 * inch, 0.75 * inch, 0.75 * inch, 0.75 * inch, 0.55 * inch],
            body_font_size=7.3,
        ),
        Spacer(1, 12),
        Image(str(total_return_chart), width=7.0 * inch, height=3.6 * inch),
        Spacer(1, 12),
        Image(str(score_chart), width=7.0 * inch, height=3.6 * inch),
    ]

    if dl_rule_delta_rows:
        story.extend(
            [
                PageBreak(),
                Paragraph("DL Rule Toggle", styles["section"]),
                Paragraph(
                    (
                        "This table compares the same DL strategies before and after disabling no-same-day "
                        "reentry. Positive deltas mean the no-rule version improved the average total return."
                    ),
                    styles["body"],
                ),
                Spacer(1, 8),
                _styled_table(
                    [
                        ["Symbol", "Strategy", "With Rule", "Without Rule", "Delta", "With DD", "Without DD"],
                        *[
                            [
                                row["symbol"],
                                row["strategy_name"],
                                _pct(row["with_rule_return"]),
                                _pct(row["without_rule_return"]),
                                _signed_pct(row["return_delta"]),
                                _pct(row["with_rule_drawdown"]),
                                _pct(row["without_rule_drawdown"]),
                            ]
                            for row in dl_rule_delta_rows[:12]
                        ],
                    ],
                    col_widths=[0.6 * inch, 1.95 * inch, 0.8 * inch, 0.85 * inch, 0.7 * inch, 0.8 * inch, 0.9 * inch],
                ),
                Spacer(1, 12),
                Image(str(dl_toggle_chart), width=7.0 * inch, height=3.5 * inch),
            ]
        )

    story.extend(
        [
            PageBreak(),
            Paragraph("Artifact Scope", styles["section"]),
            _bullet_paragraph(
                f"Rule-based comparison artifacts: {', '.join(str(path) for path in resolved_rule_artifacts)}",
                styles,
            ),
            _bullet_paragraph(
                f"ML artifacts: {', '.join(str(path) for path in resolved_ml_artifacts)}",
                styles,
            ),
            _bullet_paragraph(
                f"DL with-rule artifact: {resolved_dl_with_rule}",
                styles,
            ),
            _bullet_paragraph(
                f"DL no-rule artifact: {resolved_dl_no_rule}",
                styles,
            ),
            _bullet_paragraph(
                "Unified report score formula used for cross-branch ranking: "
                "avg_total_return + 0.5 * avg_max_drawdown + 0.1 * profitable_ratio + "
                "0.02 * min(avg_profit_factor, 3.0).",
                styles,
            ),
        ]
    )

    doc.build(story, onFirstPage=_draw_dark_page, onLaterPages=_draw_dark_page)
    summary = {
        "output_path": str(resolved_output.resolve()),
        "records_compared": len(all_records),
        "best_overall": asdict(best_overall),
        "best_dl_without_rule": asdict(best_no_rule),
        "branch_leaders": [{"branch": branch, "record": asdict(record)} for branch, record in branch_leaders],
        "dl_rule_direction": dl_rule_direction,
        "dl_avg_score_with_rule": round(avg_dl_with_rule, 6),
        "dl_avg_score_without_rule": round(avg_dl_no_rule, 6),
    }
    metadata_path = resolved_output.with_suffix(".json")
    metadata_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _load_rule_based_records(paths: list[Path]) -> list[StrategyRecord]:
    records: list[StrategyRecord] = []
    for path in paths:
        payload = _load_json(path)
        if "comparison" in payload:
            symbol = str(payload["symbol"])
            strategy_name = str(payload["strategy_name"])
            for variant in payload["comparison"]:
                label = "hourly cost-aware, no-same-day ON" if variant["no_same_day_reentry"] else "hourly cost-aware, no-same-day OFF"
                records.append(
                    _rule_variant_record(
                        market="commodities",
                        symbol=symbol,
                        strategy_name=strategy_name,
                        experiment=label,
                        results=variant["results"],
                    )
                )
            continue

        for comparison in payload.get("comparisons", []):
            market = str(comparison["market"])
            symbol = str(comparison["symbol"])
            strategy_name = str(comparison["strategy_name"])
            baseline = comparison.get("baseline")
            if baseline:
                records.append(
                    _rule_variant_record(
                        market=market,
                        symbol=symbol,
                        strategy_name=strategy_name,
                        experiment="hourly cost-aware, no-same-day OFF",
                        results=baseline["results"],
                    )
                )
            no_reentry = comparison.get("no_same_day_reentry")
            if no_reentry:
                records.append(
                    _rule_variant_record(
                        market=market,
                        symbol=symbol,
                        strategy_name=strategy_name,
                        experiment="hourly cost-aware, no-same-day ON",
                        results=no_reentry["results"],
                    )
                )
    return records


def _rule_variant_record(
    *,
    market: str,
    symbol: str,
    strategy_name: str,
    experiment: str,
    results: list[dict[str, object]],
) -> StrategyRecord:
    avg_total_return = mean(float(item["total_return"]) for item in results)
    avg_annualized_return = mean(float(item.get("annualized_return", 0.0)) for item in results)
    avg_max_drawdown = mean(float(item["max_drawdown"]) for item in results)
    avg_profit_factor = mean(float(item.get("profit_factor", 0.0)) for item in results)
    profitable_windows = sum(1 for item in results if float(item["total_return"]) > 0)
    return StrategyRecord(
        branch="Rule-based",
        experiment=experiment,
        market=market,
        symbol=symbol,
        strategy_name=strategy_name,
        windows_tested=[str(item["window"]) for item in results],
        profitable_windows=profitable_windows,
        avg_total_return=avg_total_return,
        avg_annualized_return=avg_annualized_return,
        avg_max_drawdown=avg_max_drawdown,
        avg_profit_factor=avg_profit_factor,
        report_score=_report_score(
            avg_total_return=avg_total_return,
            avg_max_drawdown=avg_max_drawdown,
            avg_profit_factor=avg_profit_factor,
            profitable_windows=profitable_windows,
            total_windows=len(results),
        ),
    )


def _load_leaderboard_records(
    artifact_paths: list[Path],
    *,
    branch: str,
    default_experiment: str,
) -> list[StrategyRecord]:
    records: list[StrategyRecord] = []
    for path in artifact_paths:
        payload = _load_json(path)
        for item in payload.get("leaderboard", []):
            windows = [str(value) for value in item.get("windows_tested", [])]
            profitable_windows = int(item.get("profitable_windows", 0))
            avg_total_return = float(item.get("avg_total_return", 0.0))
            avg_annualized_return = float(item.get("avg_annualized_return", 0.0))
            avg_max_drawdown = float(item.get("avg_max_drawdown", 0.0))
            avg_profit_factor = float(item.get("avg_profit_factor", 0.0))
            records.append(
                StrategyRecord(
                    branch=branch,
                    experiment=default_experiment,
                    market=str(item.get("market", "")),
                    symbol=str(item.get("symbol", "")),
                    strategy_name=str(item.get("strategy_name", "")),
                    windows_tested=windows,
                    profitable_windows=profitable_windows,
                    avg_total_return=avg_total_return,
                    avg_annualized_return=avg_annualized_return,
                    avg_max_drawdown=avg_max_drawdown,
                    avg_profit_factor=avg_profit_factor,
                    report_score=_report_score(
                        avg_total_return=avg_total_return,
                        avg_max_drawdown=avg_max_drawdown,
                        avg_profit_factor=avg_profit_factor,
                        profitable_windows=profitable_windows,
                        total_windows=len(windows),
                    ),
                )
            )
    return records


def _build_dl_rule_delta_rows(
    with_rule_records: list[StrategyRecord],
    without_rule_records: list[StrategyRecord],
) -> list[dict[str, object]]:
    with_rule_map = {(item.symbol, item.strategy_name): item for item in with_rule_records}
    rows: list[dict[str, object]] = []
    for item in without_rule_records:
        key = (item.symbol, item.strategy_name)
        with_rule = with_rule_map.get(key)
        if with_rule is None:
            continue
        rows.append(
            {
                "symbol": item.symbol,
                "strategy_name": item.strategy_name,
                "with_rule_return": with_rule.avg_total_return,
                "without_rule_return": item.avg_total_return,
                "return_delta": item.avg_total_return - with_rule.avg_total_return,
                "with_rule_drawdown": with_rule.avg_max_drawdown,
                "without_rule_drawdown": item.avg_max_drawdown,
            }
        )
    rows.sort(key=lambda row: float(row["return_delta"]), reverse=True)
    return rows


def _branch_leaders(records: list[StrategyRecord]) -> list[tuple[str, StrategyRecord]]:
    grouped: dict[str, list[StrategyRecord]] = {}
    for item in records:
        key = f"{item.branch} / {item.experiment}"
        grouped.setdefault(key, []).append(item)
    leaders: list[tuple[str, StrategyRecord]] = []
    for branch, items in grouped.items():
        leaders.append((branch, sorted(items, key=lambda item: item.report_score, reverse=True)[0]))
    leaders.sort(key=lambda item: item[0])
    return leaders


def _build_bar_chart(
    *,
    output_path: Path,
    records: list[StrategyRecord],
    value_attr: str,
    title: str,
    color: str,
    percent: bool,
) -> None:
    labels = [f"{item.symbol} / {item.strategy_name}" for item in records]
    values = [getattr(item, value_attr) for item in records]

    plt.style.use("dark_background")
    figure, axis = plt.subplots(figsize=(11, 5.5))
    figure.patch.set_facecolor("#0b1220")
    axis.set_facecolor("#121a2b")
    bars = axis.barh(range(len(records)), values, color=color, alpha=0.88)
    axis.set_yticks(range(len(records)))
    axis.set_yticklabels(labels, fontsize=8)
    axis.invert_yaxis()
    axis.set_title(title, fontsize=14, color="white", pad=12)
    axis.grid(axis="x", color="#24314a", alpha=0.45, linewidth=0.8)
    for index, bar in enumerate(bars):
        value = values[index]
        label = f"{value * 100:.1f}%" if percent else f"{value:.3f}"
        axis.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f" {label}",
            va="center",
            ha="left",
            fontsize=8,
            color="white",
        )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def _build_grouped_toggle_chart(output_path: Path, rows: list[dict[str, object]]) -> None:
    labels = [f"{row['symbol']} / {row['strategy_name']}" for row in rows]
    with_rule = [float(row["with_rule_return"]) * 100 for row in rows]
    without_rule = [float(row["without_rule_return"]) * 100 for row in rows]
    positions = list(range(len(rows)))

    plt.style.use("dark_background")
    figure, axis = plt.subplots(figsize=(11, 5.5))
    figure.patch.set_facecolor("#0b1220")
    axis.set_facecolor("#121a2b")
    width = 0.38
    axis.bar([value - width / 2 for value in positions], with_rule, width=width, label="Rule ON", color="#4da3ff")
    axis.bar([value + width / 2 for value in positions], without_rule, width=width, label="Rule OFF", color="#38d39f")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    axis.set_title("DL Average Total Return: No-Same-Day Reentry ON vs OFF", fontsize=14, color="white", pad=12)
    axis.set_ylabel("Average total return (%)", color="white")
    axis.grid(axis="y", color="#24314a", alpha=0.45, linewidth=0.8)
    axis.legend(facecolor="#121a2b", edgecolor="#24314a")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def _build_styles():
    base_styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base_styles["Title"],
            textColor=TEXT,
            alignment=TA_CENTER,
            fontSize=20,
            leading=24,
            spaceAfter=6,
        ),
        "section": ParagraphStyle(
            "SectionTitle",
            parent=base_styles["Heading2"],
            textColor=ACCENT,
            fontSize=13,
            leading=16,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base_styles["BodyText"],
            textColor=TEXT,
            fontSize=9.5,
            leading=13,
            spaceAfter=4,
        ),
        "bullet": ParagraphStyle(
            "BulletBody",
            parent=base_styles["BodyText"],
            textColor=TEXT,
            fontSize=9.0,
            leading=12,
            leftIndent=10,
            bulletIndent=0,
            spaceAfter=3,
        ),
    }


def _styled_table(rows: list[list[str]], *, col_widths: list[float], body_font_size: float = 8.0) -> Table:
    table = Table(rows, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
                ("TEXTCOLOR", (0, 0), (-1, 0), TEXT),
                ("BACKGROUND", (0, 1), (-1, -1), PANEL_BG),
                ("TEXTCOLOR", (0, 1), (-1, -1), TEXT),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, 0), body_font_size),
                ("FONTSIZE", (0, 1), (-1, -1), body_font_size),
                ("GRID", (0, 0), (-1, -1), 0.4, GRID),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [PANEL_BG, colors.HexColor("#0f1728")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _draw_dark_page(canvas, document) -> None:
    canvas.saveState()
    canvas.setFillColor(REPORT_BG)
    canvas.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(A4[0] - 36, 20, f"Page {document.page}")
    canvas.restoreState()


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _signed_pct(value: float) -> str:
    prefix = "+" if value >= 0 else ""
    return f"{prefix}{value * 100:.2f}%"


def _report_score(
    *,
    avg_total_return: float,
    avg_max_drawdown: float,
    avg_profit_factor: float,
    profitable_windows: int,
    total_windows: int,
) -> float:
    profitable_ratio = profitable_windows / max(1, total_windows)
    return round(
        avg_total_return + (avg_max_drawdown * 0.5) + (profitable_ratio * 0.1) + (min(avg_profit_factor, 3.0) * 0.02),
        6,
    )


def _bullet_paragraph(text: str, styles) -> Paragraph:
    return Paragraph(text, styles["bullet"], bulletText="•")
