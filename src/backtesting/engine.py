from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import pandas as pd

from src.features.macro import compute_macro
from src.features.regime import classify_regime
from src.features.risk import compute_risk
from src.features.technical import compute_technical
from src.models.features import FeatureVector, TechnicalSignal
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TradeRecord:
    entry_date: date
    exit_date: Optional[date] = None
    direction: str = "neutral"
    entry_price: float = 0.0
    exit_price: float = 0.0
    pnl_pips: float = 0.0
    regime: str = "normal"
    tech_bias: float = 0.0
    conviction: float = 0.0


@dataclass
class BacktestResult:
    trades: list[TradeRecord] = field(default_factory=list)
    daily_signals: list[dict] = field(default_factory=list)


def deterministic_signal(tech: TechnicalSignal, regime_name: str) -> tuple[str, float]:
    """Generate a deterministic trading signal without LLM.

    Returns (direction, conviction) based on technical bias and regime.
    """
    bias = tech.bias

    regime_multiplier = {
        "carry_trade": 1.2,
        "risk_off": 0.6,
        "policy_divergence": 0.8,
        "intervention": 0.3,
        "normal": 1.0,
    }
    multiplier = regime_multiplier.get(regime_name, 1.0)
    adjusted_bias = bias * multiplier

    if adjusted_bias > 0.15:
        direction = "long_jpyusd"
    elif adjusted_bias < -0.15:
        direction = "short_jpyusd"
    else:
        direction = "neutral"

    conviction = min(100.0, abs(adjusted_bias) * 100)
    return direction, conviction


def run_backtest(
    price_history: pd.DataFrame,
    settings: dict | None = None,
    min_data_points: int = 200,
    hold_period_days: int = 5,
) -> BacktestResult:
    """Replay deterministic signals over historical price data.

    No LLM calls -- purely algorithmic for speed and reproducibility.
    """
    cfg = settings or get_settings()
    result = BacktestResult()

    if len(price_history) < min_data_points:
        logger.warning("insufficient_data_for_backtest",
                       rows=len(price_history), required=min_data_points)
        return result

    active_trade: Optional[TradeRecord] = None

    for i in range(min_data_points, len(price_history)):
        window = price_history.iloc[:i + 1]
        current_date = window.index[-1]
        if hasattr(current_date, "date"):
            current_date = current_date.date()

        try:
            tech = compute_technical(window)
        except Exception:
            continue

        price = tech.current_price
        price_series = window["Close"]

        risk = compute_risk(
            current_price=price,
            atr=tech.atr,
            price_history=price_series,
            settings=cfg,
        )

        usdjpy_5d = 0.0
        if len(price_series) >= 6:
            usdjpy_5d = (price_series.iloc[-1] / price_series.iloc[-6] - 1) * 100

        regime = classify_regime(
            vix=None,
            usdjpy_price=price,
            usdjpy_5d_change=usdjpy_5d,
            yield_spread=None,
            vol_percentile=risk.volatility_percentile,
            intervention_risk=risk.intervention_risk,
            settings=cfg,
        )

        direction, conviction = deterministic_signal(tech, regime.regime.value)

        result.daily_signals.append({
            "date": current_date,
            "price": price,
            "direction": direction,
            "conviction": conviction,
            "regime": regime.regime.value,
            "tech_bias": tech.bias,
            "rsi": tech.rsi,
            "atr": tech.atr,
        })

        if active_trade is not None:
            days_held = (current_date - active_trade.entry_date).days
            if days_held >= hold_period_days:
                active_trade.exit_date = current_date
                active_trade.exit_price = price
                if active_trade.direction == "long_jpyusd":
                    active_trade.pnl_pips = (price - active_trade.entry_price) / 0.01
                elif active_trade.direction == "short_jpyusd":
                    active_trade.pnl_pips = (active_trade.entry_price - price) / 0.01
                result.trades.append(active_trade)
                active_trade = None

        if active_trade is None and direction != "neutral" and conviction > 30:
            active_trade = TradeRecord(
                entry_date=current_date,
                direction=direction,
                entry_price=price,
                regime=regime.regime.value,
                tech_bias=tech.bias,
                conviction=conviction,
            )

    if active_trade is not None:
        active_trade.exit_date = current_date
        active_trade.exit_price = price_history["Close"].iloc[-1]
        if active_trade.direction == "long_jpyusd":
            active_trade.pnl_pips = (active_trade.exit_price - active_trade.entry_price) / 0.01
        elif active_trade.direction == "short_jpyusd":
            active_trade.pnl_pips = (active_trade.entry_price - active_trade.exit_price) / 0.01
        result.trades.append(active_trade)

    return result
