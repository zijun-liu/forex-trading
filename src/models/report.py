from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from .features import FeatureVector, RegimeState
from .signals import MacroAnalysis, NewsAnalysis, SynthesizedSignal


class FXYSnapshot(BaseModel):
    price: float
    change_1d_pct: Optional[float] = None
    change_5d_pct: Optional[float] = None
    change_20d_pct: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    rsi: Optional[float] = None
    recommendation: str = ""


class AdvisoryReport(BaseModel):
    timestamp: datetime
    pair: str = "JPYUSD"
    current_price: float
    regime: RegimeState
    features: FeatureVector
    macro_analysis: MacroAnalysis
    news_analysis: NewsAnalysis
    signal: SynthesizedSignal
    fxy: Optional[FXYSnapshot] = None
    decision_log: dict = {}
