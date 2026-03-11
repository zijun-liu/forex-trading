from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.models.features import FeatureVector


def z_score(value: float, mean: float, std: float) -> float:
    if std == 0:
        return 0.0
    return (value - mean) / std


def compute_z_scores(
    feature_vector: FeatureVector,
    historical_stats: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Normalize features using historical mean/std.

    historical_stats maps feature_name -> {"mean": ..., "std": ...}.
    """
    normalized: dict[str, float] = {}

    raw = {
        "rsi": feature_vector.technical.rsi,
        "atr": feature_vector.technical.atr,
        "macd_histogram": feature_vector.technical.macd_histogram,
        "tech_bias": feature_vector.technical.bias,
    }

    macro = feature_vector.macro
    if macro.yield_spread is not None:
        raw["yield_spread"] = macro.yield_spread
    if macro.real_rate_spread is not None:
        raw["real_rate_spread"] = macro.real_rate_spread
    if macro.carry_signal is not None:
        raw["carry_signal"] = macro.carry_signal
    if macro.curve_slope is not None:
        raw["curve_slope"] = macro.curve_slope
    if macro.dxy_momentum_5d is not None:
        raw["dxy_momentum_5d"] = macro.dxy_momentum_5d
    if macro.dxy_momentum_20d is not None:
        raw["dxy_momentum_20d"] = macro.dxy_momentum_20d

    for name, value in raw.items():
        stats = historical_stats.get(name)
        if stats and "mean" in stats and "std" in stats:
            normalized[name] = round(z_score(value, stats["mean"], stats["std"]), 4)
        else:
            normalized[name] = round(value, 4)

    return normalized


def compute_stats_from_series(series: pd.Series) -> dict[str, float]:
    clean = series.dropna()
    if len(clean) < 2:
        return {"mean": 0.0, "std": 1.0}
    return {"mean": float(clean.mean()), "std": float(clean.std())}
