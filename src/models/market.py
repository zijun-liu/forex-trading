from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel


class PriceBar(BaseModel):
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class MacroSnapshot(BaseModel):
    date: date
    fed_funds_rate: Optional[float] = None
    us_cpi_yoy: Optional[float] = None
    us_10y_yield: Optional[float] = None
    us_2y_yield: Optional[float] = None
    japan_policy_rate: Optional[float] = None
    japan_cpi_yoy: Optional[float] = None
    vix: Optional[float] = None
    dxy: Optional[float] = None
    oil_price: Optional[float] = None


class MarketSnapshot(BaseModel):
    """Full market state at a point in time."""
    timestamp: datetime
    pair: str = "JPYUSD"
    price: PriceBar
    macro: MacroSnapshot
    us_10y_yield: Optional[float] = None
    us_2y_yield: Optional[float] = None
    jgb_10y_yield: Optional[float] = None
    vix: Optional[float] = None
    dxy: Optional[float] = None
    oil_price: Optional[float] = None
    cot_net_positioning: Optional[float] = None
    cot_positioning_delta_4w: Optional[float] = None
