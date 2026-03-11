from __future__ import annotations

import pandas as pd

from src.models.features import RegimeState, RegimeType

from src.utils.config import get_settings


def detect_rate_change(
    fed_rate_history: pd.Series | None,
    boj_rate_history: pd.Series | None,
    lookback_days: int = 30,
) -> bool:
    if fed_rate_history is None and boj_rate_history is None:
        return False
    if fed_rate_history is not None and len(fed_rate_history) >= 2:
        n = min(lookback_days, len(fed_rate_history))
        recent = fed_rate_history.iloc[-n:]
        if (recent.diff().fillna(0) != 0).any():
            return True
    if boj_rate_history is not None and len(boj_rate_history) >= 2:
        n = min(lookback_days, len(boj_rate_history))
        recent = boj_rate_history.iloc[-n:]
        if (recent.diff().fillna(0) != 0).any():
            return True
    return False


def classify_regime(
    vix: float | None,
    usdjpy_price: float,
    usdjpy_5d_change: float,
    yield_spread: float | None,
    vol_percentile: float,
    intervention_risk: str,
    rate_changed_recently: bool = False,
    settings: dict | None = None,
) -> RegimeState:
    reg = (settings or get_settings()).get("regime", {})
    vix_threshold = reg.get("vix_risk_off_threshold", 25)
    yield_carry_threshold = reg.get("yield_spread_carry_threshold", 3.0)
    vol_carry_max = reg.get("vol_percentile_carry_max", 50)

    if intervention_risk == "HIGH":
        return RegimeState(
            regime=RegimeType.INTERVENTION,
            confidence=0.9,
            description="High intervention risk regime",
        )

    if vix is not None and vix > vix_threshold and usdjpy_5d_change < 0:
        excess = min((vix - vix_threshold) / vix_threshold, 1.0)
        confidence = 0.5 + 0.4 * excess
        return RegimeState(
            regime=RegimeType.RISK_OFF,
            confidence=round(confidence, 2),
            description="Risk-off: elevated VIX and USD/JPY weakness",
        )

    if (
        yield_spread is not None
        and yield_spread > yield_carry_threshold
        and vol_percentile < vol_carry_max
    ):
        return RegimeState(
            regime=RegimeType.CARRY_TRADE,
            confidence=0.7,
            description="Carry trade: wide yield spread and low volatility",
        )

    if rate_changed_recently:
        return RegimeState(
            regime=RegimeType.POLICY_DIVERGENCE,
            confidence=0.6,
            description="Policy divergence: recent rate changes",
        )

    return RegimeState(
        regime=RegimeType.NORMAL,
        confidence=0.5,
        description="Normal market conditions",
    )
