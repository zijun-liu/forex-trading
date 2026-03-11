from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any

from src.backtesting.engine import BacktestResult, TradeRecord


def compute_metrics(result: BacktestResult) -> dict[str, Any]:
    trades = result.trades
    if not trades:
        return {"error": "No trades to evaluate"}

    pnls = [t.pnl_pips for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    win_rate = len(wins) / len(pnls) if pnls else 0.0
    avg_win = statistics.mean(wins) if wins else 0.0
    avg_loss = abs(statistics.mean(losses)) if losses else 0.0
    loss_rate = 1.0 - win_rate

    expectancy = avg_win * win_rate - avg_loss * loss_rate
    profit_factor = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")

    cumulative = _cumulative_pnl(pnls)
    max_dd = _max_drawdown(cumulative)
    sharpe = _sharpe_ratio(pnls)

    directions = [s.get("direction") for s in result.daily_signals]
    turnover = _turnover(directions)

    regime_perf = _performance_by_regime(trades)

    return {
        "total_trades": len(trades),
        "win_rate": round(win_rate, 4),
        "avg_win_pips": round(avg_win, 2),
        "avg_loss_pips": round(avg_loss, 2),
        "expectancy_pips": round(expectancy, 2),
        "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else "inf",
        "total_pnl_pips": round(sum(pnls), 2),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown_pips": round(max_dd, 2),
        "turnover": round(turnover, 4),
        "regime_performance": regime_perf,
    }


def _cumulative_pnl(pnls: list[float]) -> list[float]:
    cum = []
    total = 0.0
    for p in pnls:
        total += p
        cum.append(total)
    return cum


def _max_drawdown(cumulative: list[float]) -> float:
    if not cumulative:
        return 0.0
    peak = cumulative[0]
    max_dd = 0.0
    for val in cumulative:
        peak = max(peak, val)
        dd = peak - val
        max_dd = max(max_dd, dd)
    return max_dd


def _sharpe_ratio(pnls: list[float], annualization: float = 252.0) -> float:
    if len(pnls) < 2:
        return 0.0
    mean_pnl = statistics.mean(pnls)
    std_pnl = statistics.stdev(pnls)
    if std_pnl == 0:
        return 0.0
    return (mean_pnl / std_pnl) * math.sqrt(annualization)


def _turnover(directions: list[str | None]) -> float:
    if len(directions) < 2:
        return 0.0
    changes = sum(
        1 for i in range(1, len(directions))
        if directions[i] != directions[i - 1]
    )
    return changes / len(directions)


def _performance_by_regime(trades: list[TradeRecord]) -> dict[str, dict[str, Any]]:
    by_regime: dict[str, list[float]] = defaultdict(list)
    for t in trades:
        by_regime[t.regime].append(t.pnl_pips)

    result: dict[str, dict[str, Any]] = {}
    for regime, pnls in by_regime.items():
        wins = [p for p in pnls if p > 0]
        result[regime] = {
            "trades": len(pnls),
            "win_rate": round(len(wins) / len(pnls), 4) if pnls else 0,
            "total_pnl_pips": round(sum(pnls), 2),
            "sharpe": round(_sharpe_ratio(pnls), 4),
        }
    return result


def render_metrics(metrics: dict[str, Any]) -> str:
    lines = [
        "=== Backtest Results ===",
        f"Total trades:      {metrics.get('total_trades', 0)}",
        f"Win rate:          {metrics.get('win_rate', 0):.1%}",
        f"Avg win:           {metrics.get('avg_win_pips', 0):.1f} pips",
        f"Avg loss:          {metrics.get('avg_loss_pips', 0):.1f} pips",
        f"Expectancy:        {metrics.get('expectancy_pips', 0):.1f} pips/trade",
        f"Profit factor:     {metrics.get('profit_factor', 0)}",
        f"Total PnL:         {metrics.get('total_pnl_pips', 0):.1f} pips",
        f"Sharpe ratio:      {metrics.get('sharpe_ratio', 0):.4f}",
        f"Max drawdown:      {metrics.get('max_drawdown_pips', 0):.1f} pips",
        f"Turnover:          {metrics.get('turnover', 0):.1%}",
        "",
        "--- Performance by Regime ---",
    ]
    regime_perf = metrics.get("regime_performance", {})
    for regime, stats in regime_perf.items():
        lines.append(
            f"  {regime}: {stats['trades']} trades, "
            f"win rate {stats['win_rate']:.1%}, "
            f"PnL {stats['total_pnl_pips']:.1f} pips, "
            f"Sharpe {stats['sharpe']:.4f}"
        )
    return "\n".join(lines)
