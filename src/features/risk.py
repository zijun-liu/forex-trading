from __future__ import annotations

import pandas as pd

from src.models.features import RiskAssessment


def _volatility_percentile(price_history: pd.Series, atr: float, window: int = 14, lookback: int = 90) -> float:
    tr_proxy = price_history.diff().abs()
    rolling_atr = tr_proxy.rolling(window, min_periods=1).mean()
    last_90 = rolling_atr.dropna().iloc[-lookback:] if len(rolling_atr.dropna()) >= lookback else rolling_atr.dropna()
    if len(last_90) == 0:
        return 50.0
    percentile = (last_90 < atr).sum() / len(last_90) * 100
    return float(percentile)


def compute_intervention_risk(
    price: float,
    price_history: pd.Series,
    atr: float,
    settings: dict,
) -> str:
    interv = settings.get("intervention", {})
    threshold = interv.get("price_threshold", 155.0)
    vel_pct = interv.get("velocity_threshold_pct", 3.0)
    vel_window = interv.get("velocity_window_days", 5)
    vol_pct_threshold = interv.get("vol_percentile_threshold", 75)

    vol_percentile = _volatility_percentile(price_history, atr)
    price_above = price > threshold
    velocity_above = False
    if len(price_history) >= vel_window + 1:
        pct_change = (price_history.iloc[-1] / price_history.iloc[-(vel_window + 1)] - 1) * 100
        velocity_above = pct_change > vel_pct
    vol_above = vol_percentile > vol_pct_threshold

    conditions = sum([price_above, velocity_above, vol_above])
    if conditions >= 3:
        return "HIGH"
    if conditions >= 2:
        return "MEDIUM"
    return "LOW"


def compute_risk(
    current_price: float,
    atr: float,
    price_history: pd.Series,
    settings: dict,
    cot_net: float | None = None,
) -> RiskAssessment:
    interv = settings.get("intervention", {})
    risk_cfg = settings.get("risk", {})

    vol_percentile = _volatility_percentile(price_history, atr)
    intervention_risk = compute_intervention_risk(current_price, price_history, atr, settings)

    price_above = current_price > interv.get("price_threshold", 155.0)
    velocity_above = False
    if len(price_history) >= 6:
        pct_change = (price_history.iloc[-1] / price_history.iloc[-6] - 1) * 100
        velocity_above = pct_change > interv.get("velocity_threshold_pct", 3.0)
    vol_above = vol_percentile > interv.get("vol_percentile_threshold", 75)
    intervention_detected = price_above and velocity_above and vol_above

    capital = risk_cfg.get("capital", 100000)
    risk_pct = risk_cfg.get("risk_per_trade_pct", 1.0)
    stop_distance = atr * 2
    pip_value = risk_cfg.get("pip_value", 0.01)
    position_size_lots = (capital * risk_pct / 100) / (stop_distance * 100) if atr > 0 else 0.0
    stop_distance_pips = (stop_distance / pip_value) if pip_value > 0 else None

    warnings = []
    if intervention_risk == "HIGH":
        warnings.append("High intervention risk")
    if intervention_detected:
        warnings.append("Intervention conditions detected")
    if position_size_lots > 1.0:
        warnings.append("Extreme positioning")
    if vol_percentile > 90:
        warnings.append("High volatility")

    return RiskAssessment(
        intervention_risk=intervention_risk,
        volatility_percentile=round(vol_percentile, 2),
        position_size_lots=round(position_size_lots, 4),
        stop_distance_pips=round(stop_distance_pips, 2) if stop_distance_pips is not None else None,
        warnings=warnings,
    )
