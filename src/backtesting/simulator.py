from __future__ import annotations

import argparse

import yfinance as yf
from rich.console import Console

from src.backtesting.engine import run_backtest
from src.backtesting.metrics import compute_metrics, render_metrics
from src.utils.logger import get_logger

logger = get_logger(__name__)
console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run backtest on USD/JPY historical data")
    parser.add_argument("--period", default="5y", help="Data period (default: 5y)")
    parser.add_argument("--hold-days", type=int, default=5, help="Trade holding period in days")
    parser.add_argument("--json", action="store_true", help="Output JSON metrics")
    args = parser.parse_args()

    console.print(f"[dim]Fetching {args.period} of USD/JPY data...[/dim]")
    df = yf.Ticker("JPY=X").history(period=args.period, interval="1d")

    if df.empty:
        console.print("[red]No data fetched. Check network connection.[/red]")
        return

    console.print(f"[dim]Running backtest on {len(df)} days of data...[/dim]")
    result = run_backtest(df, hold_period_days=args.hold_days)
    metrics = compute_metrics(result)

    if args.json:
        import json
        print(json.dumps(metrics, indent=2, default=str))
    else:
        console.print()
        console.print(render_metrics(metrics))
        console.print()


if __name__ == "__main__":
    main()
