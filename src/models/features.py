from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel


class TrendDirection(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class TechnicalSignal(BaseModel):
    trend: TrendDirection
    trend_strength: float  # 0-1

    sma_20: float
    sma_50: float
    sma_200: float
    macd: float
    macd_signal: float
    macd_histogram: float
    rsi: float
    atr: float
    bb_upper: float
    bb_middle: float
    bb_lower: float

    support: Optional[float] = None
    resistance: Optional[float] = None
    current_price: float

    bias: float  # -1 (bearish) to +1 (bullish), derived deterministically


class MacroFeatures(BaseModel):
    yield_spread: Optional[float] = None      # US10Y - JGB10Y
    real_rate_spread: Optional[float] = None   # (US_nom - US_CPI) - (JP_nom - JP_CPI)
    carry_signal: Optional[float] = None       # rate_diff - vol_penalty
    curve_slope: Optional[float] = None        # US10Y - US2Y
    dxy_momentum_5d: Optional[float] = None    # DXY % change 5d
    dxy_momentum_20d: Optional[float] = None   # DXY % change 20d
    oil_jpy_change: Optional[float] = None     # oil_usd * usdjpy pct change


class RiskAssessment(BaseModel):
    intervention_risk: str  # LOW / MEDIUM / HIGH
    volatility_percentile: float  # 0-100
    position_size_lots: float
    stop_distance_pips: Optional[float] = None
    warnings: list[str] = []


class RegimeType(str, Enum):
    CARRY_TRADE = "carry_trade"
    RISK_OFF = "risk_off"
    POLICY_DIVERGENCE = "policy_divergence"
    INTERVENTION = "intervention"
    NORMAL = "normal"


class RegimeState(BaseModel):
    regime: RegimeType
    confidence: float  # 0-1
    description: str


class FeatureVector(BaseModel):
    """All deterministic features, optionally z-score normalized."""
    technical: TechnicalSignal
    macro: MacroFeatures
    risk: RiskAssessment
    regime: RegimeState
    normalized: dict[str, float] = {}  # feature_name -> z-score
