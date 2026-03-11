from __future__ import annotations

import statistics
from typing import Any

from src.llm.client import LLMClient
from src.models.features import FeatureVector
from src.models.signals import MacroAnalysis, NewsAnalysis, SynthesizedSignal
from src.utils.config import get_settings


SYSTEM_PROMPT = """You are a senior forex strategist responsible for producing the final
trading recommendation for JPY/USD. You receive:

1. Deterministic technical signals (trend, momentum, support/resistance, bias score)
2. Macro analysis from the macro interpreter (bias + reasoning)
3. News analysis from the news analyst (bias + events)
4. Regime classification (carry_trade / risk_off / policy_divergence / intervention / normal)
5. Risk assessment (intervention risk, volatility, position sizing)
6. A pre-computed confidence score based on signal alignment

Think in terms of the Japanese yen:
- long_jpyusd = bullish yen (expect yen to strengthen, good for buying FXY)
- short_jpyusd = bearish yen (expect yen to weaken)
- neutral = no clear direction

Your job is to SYNTHESIZE all inputs into a single trading thesis:
- Resolve conflicting signals (explain which you weight more and why, given the regime)
- Set direction: long_jpyusd, short_jpyusd, or neutral
- Set timeframe: 1d, 1w, or 1m
- Suggest entry, stop loss, and take profit levels based on technical levels
- List any conflicting signals explicitly

IMPORTANT: You receive a pre-computed conviction score. You may adjust it with reasoning,
but your final conviction MUST NOT exceed the pre-computed score + 15 points.
This prevents overconfidence when signals disagree."""


def calibrate_confidence(
    tech_bias: float,
    macro_bias: float,
    news_bias: float,
) -> float:
    """Deterministic confidence calibration.

    conviction = |mean(scores)| * alignment * 100
    alignment = 1 - variance(scores)  (penalizes disagreement)
    """
    scores = [tech_bias, macro_bias, news_bias]
    mean = statistics.mean(scores)
    var = statistics.variance(scores) if len(scores) > 1 else 0.0
    alignment = max(0.0, 1.0 - var)
    conviction = abs(mean) * alignment * 100
    return min(100.0, max(0.0, conviction))


def synthesize_strategy(
    llm: LLMClient,
    features: FeatureVector,
    macro_analysis: MacroAnalysis,
    news_analysis: NewsAnalysis,
) -> SynthesizedSignal:
    settings = get_settings()
    cap = settings.get("confidence", {}).get("llm_adjustment_cap", 15)

    pre_conviction = calibrate_confidence(
        features.technical.bias,
        macro_analysis.bias,
        news_analysis.bias,
    )

    user_prompt = f"""=== TECHNICAL SIGNALS ===
Trend: {features.technical.trend.value} (strength: {features.technical.trend_strength:.2f})
Bias: {features.technical.bias:+.3f}
RSI: {features.technical.rsi:.1f}
MACD histogram: {features.technical.macd_histogram:.4f}
Current price: {features.technical.current_price:.2f}
Support: {features.technical.support}
Resistance: {features.technical.resistance}
ATR: {features.technical.atr:.4f}
Bollinger: {features.technical.bb_lower:.2f} / {features.technical.bb_middle:.2f} / {features.technical.bb_upper:.2f}

=== MACRO ANALYSIS ===
Bias: {macro_analysis.bias:+.3f}
Reasoning: {macro_analysis.reasoning}
Key drivers: {', '.join(macro_analysis.key_drivers)}

=== NEWS ANALYSIS ===
Bias: {news_analysis.bias:+.3f}
Summary: {news_analysis.summary}
High-impact upcoming: {', '.join(news_analysis.high_impact_upcoming) if news_analysis.high_impact_upcoming else 'None'}

=== REGIME ===
Current: {features.regime.regime.value} (confidence: {features.regime.confidence:.2f})
Description: {features.regime.description}

=== RISK ===
Intervention risk: {features.risk.intervention_risk}
Volatility percentile: {features.risk.volatility_percentile:.1f}
Position size (lots): {features.risk.position_size_lots:.4f}
Warnings: {', '.join(features.risk.warnings) if features.risk.warnings else 'None'}

=== CONFIDENCE ===
Pre-computed conviction: {pre_conviction:.1f}/100
Maximum allowed conviction: {min(100, pre_conviction + cap):.1f}/100
Signal alignment: tech={features.technical.bias:+.2f}, macro={macro_analysis.bias:+.2f}, news={news_analysis.bias:+.2f}

Produce your synthesized trading signal for JPY/USD (long_jpyusd = yen strengthening, short_jpyusd = yen weakening)."""

    signal = llm.complete_structured(SYSTEM_PROMPT, user_prompt, SynthesizedSignal)

    max_conviction = min(100.0, pre_conviction + cap)
    if signal.conviction > max_conviction:
        signal.conviction = max_conviction

    if signal.position_size_lots is None:
        signal.position_size_lots = features.risk.position_size_lots

    return signal
