from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.models.report import AdvisoryReport
from src.pipeline import ForexAdvisorPipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)
console = Console()

_REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"


def render_report(report: AdvisoryReport) -> None:
    sig = report.signal
    regime = report.regime

    direction_colors = {
        "long_jpyusd": "green",
        "short_jpyusd": "red",
        "neutral": "yellow",
    }
    color = direction_colors.get(sig.direction.value, "white")

    # Header
    console.print()
    console.print(Panel(
        f"[bold]{report.pair}[/bold]  |  "
        f"Price: [cyan]{report.current_price:.2f}[/cyan]  |  "
        f"Regime: [magenta]{regime.regime.value}[/magenta] ({regime.confidence:.0%})  |  "
        f"{report.timestamp.strftime('%Y-%m-%d %H:%M')}",
        title="[bold]JPY/USD Forex Advisor[/bold]",
        border_style="blue",
    ))

    # Signal
    signal_text = Text()
    signal_text.append(f"  {sig.direction.value.upper()}", style=f"bold {color}")
    signal_text.append(f"  Conviction: {sig.conviction:.0f}/100", style="bold")
    signal_text.append(f"  Timeframe: {sig.timeframe.value}", style="dim")

    console.print(Panel(signal_text, title="Signal", border_style=color))

    # Levels
    if any([sig.entry_price, sig.stop_loss, sig.take_profit]):
        levels = Table(show_header=False, box=None, padding=(0, 2))
        levels.add_column(style="dim")
        levels.add_column(style="bold")
        if sig.entry_price:
            levels.add_row("Entry", f"{sig.entry_price:.2f}")
        if sig.stop_loss:
            levels.add_row("Stop Loss", f"[red]{sig.stop_loss:.2f}[/red]")
        if sig.take_profit:
            levels.add_row("Take Profit", f"[green]{sig.take_profit:.2f}[/green]")
        if sig.position_size_lots:
            levels.add_row("Position Size", f"{sig.position_size_lots:.4f} lots")
        console.print(Panel(levels, title="Trade Levels", border_style="cyan"))

    # Signal Alignment
    dl = report.decision_log
    alignment = Table(show_header=True, box=None, padding=(0, 2))
    alignment.add_column("Source", style="dim")
    alignment.add_column("Bias", justify="right")
    for source, key in [("Technical", "tech_bias"), ("Macro", "macro_bias"), ("News", "news_bias")]:
        val = dl.get(key, 0)
        c = "green" if val > 0.1 else ("red" if val < -0.1 else "yellow")
        alignment.add_row(source, f"[{c}]{val:+.3f}[/{c}]")
    alignment.add_row("", "")
    alignment.add_row("Pre-computed conviction", f"{dl.get('pre_conviction', 0):.1f}")
    alignment.add_row("Final conviction", f"[bold]{dl.get('final_conviction', 0):.1f}[/bold]")
    console.print(Panel(alignment, title="Signal Alignment", border_style="yellow"))

    # Reasoning
    console.print(Panel(sig.reasoning, title="Reasoning", border_style="white"))

    # Conflicts
    if sig.conflicting_signals:
        conflict_text = "\n".join(f"  - {c}" for c in sig.conflicting_signals)
        console.print(Panel(conflict_text, title="Conflicting Signals", border_style="red"))

    # Risk Warnings
    if report.features.risk.warnings:
        warnings_text = "\n".join(f"  - {w}" for w in report.features.risk.warnings)
        console.print(Panel(warnings_text, title="Risk Warnings", border_style="red"))

    # Macro Analysis Summary
    console.print(Panel(
        f"Bias: {report.macro_analysis.bias:+.3f}\n\n"
        f"{report.macro_analysis.reasoning}\n\n"
        f"Key drivers: {', '.join(report.macro_analysis.key_drivers)}",
        title="Macro Analysis",
        border_style="blue",
    ))

    # News Summary
    console.print(Panel(
        f"Bias: {report.news_analysis.bias:+.3f}\n\n"
        f"{report.news_analysis.summary}",
        title="News Analysis",
        border_style="magenta",
    ))

    # FXY ETF
    if report.fxy:
        fxy = report.fxy
        fxy_table = Table(show_header=False, box=None, padding=(0, 2))
        fxy_table.add_column(style="dim")
        fxy_table.add_column(style="bold")
        fxy_table.add_row("Price", f"${fxy.price:.2f}")
        if fxy.change_1d_pct is not None:
            c = "green" if fxy.change_1d_pct > 0 else "red"
            fxy_table.add_row("1-Day Change", f"[{c}]{fxy.change_1d_pct:+.2f}%[/{c}]")
        if fxy.change_5d_pct is not None:
            c = "green" if fxy.change_5d_pct > 0 else "red"
            fxy_table.add_row("5-Day Change", f"[{c}]{fxy.change_5d_pct:+.2f}%[/{c}]")
        if fxy.change_20d_pct is not None:
            c = "green" if fxy.change_20d_pct > 0 else "red"
            fxy_table.add_row("20-Day Change", f"[{c}]{fxy.change_20d_pct:+.2f}%[/{c}]")
        if fxy.sma_20 is not None:
            fxy_table.add_row("SMA 20", f"${fxy.sma_20:.2f}")
        if fxy.sma_50 is not None:
            fxy_table.add_row("SMA 50", f"${fxy.sma_50:.2f}")
        if fxy.rsi is not None:
            fxy_table.add_row("RSI (14)", f"{fxy.rsi:.1f}")
        fxy_table.add_row("", "")
        rec_color = "green" if "BUY" in fxy.recommendation else ("red" if "AVOID" in fxy.recommendation or "SELL" in fxy.recommendation else "yellow")
        fxy_table.add_row("Action", f"[bold {rec_color}]{fxy.recommendation}[/bold {rec_color}]")
        console.print(Panel(fxy_table, title="FXY ETF (Invesco CurrencyShares Japanese Yen Trust)", border_style="cyan"))

    console.print()


def save_report_markdown(report: AdvisoryReport) -> Path:
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = report.timestamp.strftime("%Y-%m-%d_%H%M")
    path = _REPORTS_DIR / f"report_{ts}.md"

    sig = report.signal
    regime = report.regime
    dl = report.decision_log

    lines = [
        f"# JPY/USD Advisory Report",
        f"",
        f"**Date:** {report.timestamp.strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"**Price:** {report.current_price:.2f}  ",
        f"**Regime:** {regime.regime.value} ({regime.confidence:.0%})  ",
        f"",
        f"---",
        f"",
        f"## Signal",
        f"",
        f"| | |",
        f"|---|---|",
        f"| **Direction** | {sig.direction.value.upper()} |",
        f"| **Conviction** | {sig.conviction:.0f} / 100 |",
        f"| **Timeframe** | {sig.timeframe.value} |",
    ]

    if sig.entry_price:
        lines.append(f"| **Entry** | {sig.entry_price:.2f} |")
    if sig.stop_loss:
        lines.append(f"| **Stop Loss** | {sig.stop_loss:.2f} |")
    if sig.take_profit:
        lines.append(f"| **Take Profit** | {sig.take_profit:.2f} |")
    if sig.position_size_lots:
        lines.append(f"| **Position Size** | {sig.position_size_lots:.4f} lots |")

    lines += [
        f"",
        f"---",
        f"",
        f"## Signal Alignment",
        f"",
        f"| Source | Bias |",
        f"|--------|------|",
        f"| Technical | {dl.get('tech_bias', 0):+.3f} |",
        f"| Macro | {dl.get('macro_bias', 0):+.3f} |",
        f"| News | {dl.get('news_bias', 0):+.3f} |",
        f"",
        f"- **Pre-computed conviction:** {dl.get('pre_conviction', 0):.1f}",
        f"- **Final conviction:** {dl.get('final_conviction', 0):.1f}",
        f"",
        f"---",
        f"",
        f"## Reasoning",
        f"",
        sig.reasoning,
        f"",
    ]

    if sig.conflicting_signals:
        lines += [
            f"---",
            f"",
            f"## Conflicting Signals",
            f"",
        ]
        for c in sig.conflicting_signals:
            lines.append(f"- {c}")
        lines.append("")

    if report.features.risk.warnings:
        lines += [
            f"---",
            f"",
            f"## Risk Warnings",
            f"",
        ]
        for w in report.features.risk.warnings:
            lines.append(f"- {w}")
        lines.append("")

    lines += [
        f"---",
        f"",
        f"## Macro Analysis",
        f"",
        f"**Bias:** {report.macro_analysis.bias:+.3f}",
        f"",
        report.macro_analysis.reasoning,
        f"",
        f"**Key drivers:** {', '.join(report.macro_analysis.key_drivers)}",
        f"",
        f"---",
        f"",
        f"## News Analysis",
        f"",
        f"**Bias:** {report.news_analysis.bias:+.3f}",
        f"",
        report.news_analysis.summary,
        f"",
    ]

    if report.news_analysis.events:
        lines += [
            f"### Events",
            f"",
            f"| Event | Type | Impact | Bias |",
            f"|-------|------|--------|------|",
        ]
        for evt in report.news_analysis.events:
            lines.append(
                f"| {evt.title} | {evt.event_type.value} | {evt.impact_score:.2f} | {evt.directional_bias:+.2f} |"
            )
        lines.append("")

    if report.news_analysis.high_impact_upcoming:
        lines += [
            f"### Upcoming High-Impact Events",
            f"",
        ]
        for evt in report.news_analysis.high_impact_upcoming:
            lines.append(f"- {evt}")
        lines.append("")

    lines += [
        f"---",
        f"",
        f"## Technical Snapshot",
        f"",
        f"| Indicator | Value |",
        f"|-----------|-------|",
        f"| SMA 20 | {report.features.technical.sma_20:.2f} |",
        f"| SMA 50 | {report.features.technical.sma_50:.2f} |",
        f"| SMA 200 | {report.features.technical.sma_200:.2f} |",
        f"| RSI (14) | {report.features.technical.rsi:.1f} |",
        f"| ATR (14) | {report.features.technical.atr:.4f} |",
        f"| MACD Histogram | {report.features.technical.macd_histogram:.4f} |",
        f"| Bollinger Upper | {report.features.technical.bb_upper:.2f} |",
        f"| Bollinger Lower | {report.features.technical.bb_lower:.2f} |",
        f"| Trend | {report.features.technical.trend.value} |",
        f"| Trend Strength | {report.features.technical.trend_strength:.2f} |",
    ]
    if report.features.technical.support:
        lines.append(f"| Support | {report.features.technical.support:.2f} |")
    if report.features.technical.resistance:
        lines.append(f"| Resistance | {report.features.technical.resistance:.2f} |")

    if report.fxy:
        fxy = report.fxy
        lines += [
            f"",
            f"---",
            f"",
            f"## FXY ETF (Invesco CurrencyShares Japanese Yen Trust)",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Price | ${fxy.price:.2f} |",
        ]
        if fxy.change_1d_pct is not None:
            lines.append(f"| 1-Day Change | {fxy.change_1d_pct:+.2f}% |")
        if fxy.change_5d_pct is not None:
            lines.append(f"| 5-Day Change | {fxy.change_5d_pct:+.2f}% |")
        if fxy.change_20d_pct is not None:
            lines.append(f"| 20-Day Change | {fxy.change_20d_pct:+.2f}% |")
        if fxy.sma_20 is not None:
            lines.append(f"| SMA 20 | ${fxy.sma_20:.2f} |")
        if fxy.sma_50 is not None:
            lines.append(f"| SMA 50 | ${fxy.sma_50:.2f} |")
        if fxy.rsi is not None:
            lines.append(f"| RSI (14) | {fxy.rsi:.1f} |")
        lines += [
            f"",
            f"**Recommendation:** {fxy.recommendation}",
            f"",
        ]

    lines += [
        f"---",
        f"",
        f"## Regime",
        f"",
        f"**{regime.regime.value}** (confidence: {regime.confidence:.0%})",
        f"",
        regime.description,
        f"",
    ]

    path.write_text("\n".join(lines))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="JPY/USD Forex Financial Advisor Agent"
    )
    parser.add_argument(
        "--json", action="store_true", help="Output raw JSON instead of rich format"
    )
    parser.add_argument(
        "--monitor", action="store_true",
        help="Run in monitoring mode (refresh every 4 hours)"
    )
    args = parser.parse_args()

    if args.monitor:
        _run_monitor()
        return

    pipeline = ForexAdvisorPipeline()
    try:
        console.print("[dim]Running analysis pipeline...[/dim]")
        report = pipeline.run()

        if args.json:
            print(report.model_dump_json(indent=2))
        else:
            render_report(report)

        saved = save_report_markdown(report)
        console.print(f"[dim]Report saved to {saved}[/dim]")
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
    except Exception as e:
        console.print(f"[red]Pipeline error: {e}[/red]")
        logger.exception("pipeline_error", error=str(e))
        sys.exit(1)
    finally:
        pipeline.close()


def _run_monitor() -> None:
    from apscheduler.schedulers.blocking import BlockingScheduler

    scheduler = BlockingScheduler()
    console.print("[bold]Monitor mode:[/bold] running every 4 hours. Press Ctrl+C to stop.")

    def _job() -> None:
        console.print(f"\n[dim]--- Run at {datetime.now().isoformat()} ---[/dim]")
        pipeline = ForexAdvisorPipeline()
        try:
            report = pipeline.run()
            render_report(report)
            saved = save_report_markdown(report)
            console.print(f"[dim]Report saved to {saved}[/dim]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
        finally:
            pipeline.close()

    _job()  # run immediately
    scheduler.add_job(_job, "interval", hours=4)
    try:
        scheduler.start()
    except KeyboardInterrupt:
        console.print("\n[dim]Monitor stopped.[/dim]")


if __name__ == "__main__":
    main()
