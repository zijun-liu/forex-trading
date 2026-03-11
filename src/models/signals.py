from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel


class Direction(str, Enum):
    LONG_JPYUSD = "long_jpyusd"
    SHORT_JPYUSD = "short_jpyusd"
    NEUTRAL = "neutral"


class Timeframe(str, Enum):
    DAY = "1d"
    WEEK = "1w"
    MONTH = "1m"


class EventType(str, Enum):
    MONETARY_POLICY = "monetary_policy"
    ECONOMIC_DATA = "economic_data"
    GEOPOLITICAL = "geopolitical"
    INTERVENTION = "intervention"
    RISK_SENTIMENT = "risk_sentiment"


class NewsEvent(BaseModel):
    title: str
    source: str
    event_type: EventType
    impact_score: float  # 0-1
    directional_bias: float  # -1 to +1
    summary: str


class MacroAnalysis(BaseModel):
    bias: float  # -1 (bearish USD) to +1 (bullish USD)
    reasoning: str
    key_drivers: list[str]


class NewsAnalysis(BaseModel):
    bias: float  # -1 to +1
    events: list[NewsEvent]
    high_impact_upcoming: list[str]
    summary: str


class SynthesizedSignal(BaseModel):
    direction: Direction
    conviction: float  # 0-100
    timeframe: Timeframe
    reasoning: str
    conflicting_signals: list[str]
    regime: str
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size_lots: Optional[float] = None
