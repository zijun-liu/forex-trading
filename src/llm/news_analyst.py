from __future__ import annotations

from pydantic import BaseModel

from src.llm.client import LLMClient
from src.models.signals import EventType, NewsAnalysis, NewsEvent
from src.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a forex news analyst specializing in JPY/USD.
You receive pre-classified news headlines from financial RSS feeds.

Think in terms of the Japanese yen's strength:
- Positive bias = yen strengthening (good for buying yen / FXY)
- Negative bias = yen weakening (bad for buying yen / FXY)

Your job is to:
1. Provide an overall news bias for JPY/USD from -1.0 (yen weakening) to +1.0 (yen strengthening)
2. Write a 2-3 sentence summary of the key news themes
3. List up to 3 upcoming high-impact events (as short strings)
4. List the top 3 most market-moving headlines with their impact (0-1) and bias (-1 to +1, positive = yen bullish)

Keep event titles SHORT (under 60 chars). Use only these event_type values:
monetary_policy, economic_data, geopolitical, intervention, risk_sentiment"""


class _SimpleEvent(BaseModel):
    title: str
    event_type: str
    impact: float
    bias: float


class _SimpleNewsResponse(BaseModel):
    bias: float
    summary: str
    high_impact_upcoming: list[str]
    top_events: list[_SimpleEvent]


def analyze_news(
    llm: LLMClient,
    classified_news: list[dict],
    calendar_events: list[dict] | None = None,
) -> NewsAnalysis:
    news_str = ""
    for i, item in enumerate(classified_news[:10], 1):
        title = item.get("title", "N/A")[:80]
        news_str += f"{i}. [{item.get('event_type', 'other')}] {title}\n"

    calendar_str = ""
    if calendar_events:
        for evt in calendar_events[:5]:
            calendar_str += (
                f"- {evt.get('date', '')}: {evt.get('event', '')} "
                f"({evt.get('country', '')})\n"
            )

    user_prompt = f"""Recent headlines:
{news_str if news_str else "No recent news available."}

Upcoming events:
{calendar_str if calendar_str else "No calendar data available."}

Provide your news analysis for JPY/USD (positive = yen strengthening)."""

    simple = llm.complete_structured(SYSTEM_PROMPT, user_prompt, _SimpleNewsResponse)

    valid_types = {e.value for e in EventType}
    events = []
    for evt in simple.top_events[:5]:
        etype = evt.event_type if evt.event_type in valid_types else "risk_sentiment"
        events.append(NewsEvent(
            title=evt.title[:80],
            source="aggregated",
            event_type=EventType(etype),
            impact_score=max(0.0, min(1.0, evt.impact)),
            directional_bias=max(-1.0, min(1.0, evt.bias)),
            summary=evt.title,
        ))

    return NewsAnalysis(
        bias=max(-1.0, min(1.0, simple.bias)),
        events=events,
        high_impact_upcoming=simple.high_impact_upcoming[:5],
        summary=simple.summary,
    )
