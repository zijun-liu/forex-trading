from __future__ import annotations

import pandas as pd

from src.models.features import MacroFeatures


def compute_macro(
    us_10y: float | None = None,
    us_2y: float | None = None,
    jgb_10y: float | None = None,
    us_cpi: float | None = None,
    jp_cpi: float | None = None,
    us_rate: float | None = None,
    jp_rate: float | None = None,
    dxy_history: pd.Series | None = None,
    oil_usdjpy_history: pd.DataFrame | None = None,
    realized_vol: float = 0,
) -> MacroFeatures:
    yield_spread = (us_10y - jgb_10y) if us_10y is not None and jgb_10y is not None else None
    real_rate_spread = (
        (us_rate - us_cpi) - (jp_rate - jp_cpi)
        if all(x is not None for x in (us_rate, us_cpi, jp_rate, jp_cpi))
        else None
    )
    carry_signal = (
        (us_rate - jp_rate) - realized_vol
        if us_rate is not None and jp_rate is not None
        else None
    )
    curve_slope = (us_10y - us_2y) if us_10y is not None and us_2y is not None else None

    dxy_momentum_5d = None
    dxy_momentum_20d = None
    if dxy_history is not None and len(dxy_history) >= 6:
        dxy_momentum_5d = float((dxy_history.iloc[-1] / dxy_history.iloc[-6]) - 1)
    if dxy_history is not None and len(dxy_history) >= 21:
        dxy_momentum_20d = float((dxy_history.iloc[-1] / dxy_history.iloc[-21]) - 1)

    oil_jpy_change = None
    if oil_usdjpy_history is not None and "oil" in oil_usdjpy_history.columns and "usdjpy" in oil_usdjpy_history.columns:
        product = oil_usdjpy_history["oil"] * oil_usdjpy_history["usdjpy"]
        if len(product) >= 2:
            oil_jpy_change = float(product.pct_change().iloc[-1])

    return MacroFeatures(
        yield_spread=yield_spread,
        real_rate_spread=real_rate_spread,
        carry_signal=carry_signal,
        curve_slope=curve_slope,
        dxy_momentum_5d=dxy_momentum_5d,
        dxy_momentum_20d=dxy_momentum_20d,
        oil_jpy_change=oil_jpy_change,
    )
