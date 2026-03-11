from __future__ import annotations

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands

from src.models.features import TechnicalSignal, TrendDirection


def compute_technical(df: pd.DataFrame, current_price: float | None = None) -> TechnicalSignal:
    if current_price is None:
        current_price = float(df["Close"].iloc[-1])

    sma_20 = SMAIndicator(close=df["Close"], window=20).sma_indicator()
    sma_50 = SMAIndicator(close=df["Close"], window=50).sma_indicator()
    sma_200 = SMAIndicator(close=df["Close"], window=200).sma_indicator()
    macd = MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9)
    rsi = RSIIndicator(close=df["Close"], window=14).rsi()
    atr = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=14).average_true_range()
    bb = BollingerBands(close=df["Close"], window=20, window_dev=2)

    support = df["Low"].rolling(20).min().iloc[-1] if len(df) >= 20 else None
    resistance = df["High"].rolling(20).max().iloc[-1] if len(df) >= 20 else None

    s20 = float(sma_20.iloc[-1])
    s50 = float(sma_50.iloc[-1])
    s200 = float(sma_200.iloc[-1])
    atr_val = float(atr.iloc[-1])
    macd_hist = float(macd.macd_diff().iloc[-1])
    rsi_val = float(rsi.iloc[-1])

    if s20 > s50 > s200:
        trend = TrendDirection.BULLISH
    elif s20 < s50 < s200:
        trend = TrendDirection.BEARISH
    else:
        trend = TrendDirection.NEUTRAL

    trend_strength = min(abs(s20 - s50) / atr_val, 1.0) if atr_val > 0 else 0.0

    rsi_score = (rsi_val - 50) / 50
    trend_score = 1.0 if trend == TrendDirection.BULLISH else (-1.0 if trend == TrendDirection.BEARISH else 0.0)
    macd_score = (1.0 if macd_hist > 0 else -1.0) * min(abs(macd_hist) / atr_val, 1.0) if atr_val > 0 else 0.0
    bias = max(-1.0, min(1.0, (rsi_score + trend_score + macd_score) / 3))

    return TechnicalSignal(
        trend=trend,
        trend_strength=round(trend_strength, 6),
        sma_20=s20,
        sma_50=s50,
        sma_200=s200,
        macd=float(macd.macd().iloc[-1]),
        macd_signal=float(macd.macd_signal().iloc[-1]),
        macd_histogram=macd_hist,
        rsi=rsi_val,
        atr=atr_val,
        bb_upper=float(bb.bollinger_hband().iloc[-1]),
        bb_middle=float(bb.bollinger_mavg().iloc[-1]),
        bb_lower=float(bb.bollinger_lband().iloc[-1]),
        support=support,
        resistance=resistance,
        current_price=current_price,
        bias=round(bias, 6),
    )
