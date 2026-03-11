from __future__ import annotations

from typing import Any

from src.llm.client import LLMClient
from src.models.signals import MacroAnalysis

SYSTEM_PROMPT = """You are a senior macro analyst specializing in JPY/USD forex trading.
You receive pre-computed macro features (yield spreads, real rate differentials, carry signals,
yield curve shape, DXY momentum) along with recent trend context.

Your job is to INTERPRET what these numbers mean for JPY/USD direction over the next 1-4 weeks.
Think in terms of the Japanese yen's strength:
- Positive bias = yen strengthening (bullish JPY/USD, good for buying yen / FXY)
- Negative bias = yen weakening (bearish JPY/USD, bad for buying yen / FXY)

Consider:
- How current values compare to historical norms (z-scores provided)
- Whether trends are accelerating or reversing
- Implications of central bank policy divergence (Fed vs BoJ)
- Carry trade attractiveness given current volatility
- What recent changes in yield spreads signal

Output a directional bias from -1.0 (strongly bearish JPY = yen weakening)
to +1.0 (strongly bullish JPY = yen strengthening), with reasoning."""


def interpret_macro(
    llm: LLMClient,
    macro_features: dict[str, Any],
    normalized_features: dict[str, float],
    trend_context: list[dict[str, Any]],
) -> MacroAnalysis:
    context_str = ""
    for trend in trend_context:
        if trend.get("data_points", 0) > 0:
            context_str += (
                f"- {trend['feature']}: {trend.get('direction', 'flat')} over "
                f"{trend['data_points']} days, from {trend.get('earliest', '?')} to "
                f"{trend.get('latest', '?')} (change: {trend.get('change', 0):.4f})\n"
            )

    user_prompt = f"""Current macro features:
{_format_dict(macro_features)}

Z-score normalized values:
{_format_dict(normalized_features)}

Recent trends (from market memory):
{context_str if context_str else "No historical context available yet."}

Provide your macro analysis for JPY/USD (positive = yen strengthening)."""

    return llm.complete_structured(SYSTEM_PROMPT, user_prompt, MacroAnalysis)


def _format_dict(d: dict) -> str:
    lines = []
    for k, v in d.items():
        if v is not None:
            lines.append(f"  {k}: {v}")
    return "\n".join(lines) if lines else "  (no data)"
