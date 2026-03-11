from __future__ import annotations

import feedparser
from src.utils.cache import cached_get
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

RSS_TTL = 1800

_MONETARY_KEYWORDS = {"rate", "rates", "boj", "fed", "fomc", "ycc", "yield curve", "policy rate", "interest rate"}
_ECONOMIC_KEYWORDS = {"cpi", "gdp", "employment", "jobs", "retail sales", "industrial", "trade balance", "pmi", "nfp"}
_GEOPOLITICAL_KEYWORDS = {"war", "conflict", "tariff", "sanctions", "election", "geopolitical"}
_INTERVENTION_KEYWORDS = {"intervene", "intervention", "mof", "ministry of finance", "verbal intervention"}
_RISK_KEYWORDS = {"risk", "sentiment", "fear", "safe haven", "aversion"}


class NewsFetcher:
    def __init__(self) -> None:
        settings = get_settings()
        feeds = settings.get("news", {}).get("rss_feeds", [])
        self._feeds = [f for f in feeds if isinstance(f, dict) and "url" in f]
        self._log = logger

    def fetch_all(self) -> list[dict]:
        items: list[dict] = []
        for feed_config in self._feeds:
            url = feed_config.get("url", "")
            name = feed_config.get("name", "unknown")
            key = f"rss:{url}"

            def _fetch() -> list[dict]:
                result: list[dict] = []
                try:
                    parsed = feedparser.parse(url)
                    for entry in parsed.entries:
                        result.append({
                            "title": entry.get("title", ""),
                            "summary": entry.get("summary", ""),
                            "published": entry.get("published", ""),
                            "source": name,
                            "link": entry.get("link", ""),
                        })
                    return result
                except Exception as e:
                    self._log.warning("rss_fetch_error", url=url, error=str(e))
                return result

            cached = cached_get(key, _fetch, ttl_seconds=RSS_TTL)
            items.extend(cached)
        return items

    def classify_event(self, title: str, summary: str) -> str:
        text = f"{title} {summary}".lower()
        if any(kw in text for kw in _MONETARY_KEYWORDS):
            return "monetary_policy"
        if any(kw in text for kw in _ECONOMIC_KEYWORDS):
            return "economic_data"
        if any(kw in text for kw in _GEOPOLITICAL_KEYWORDS):
            return "geopolitical"
        if any(kw in text for kw in _INTERVENTION_KEYWORDS):
            return "intervention"
        if any(kw in text for kw in _RISK_KEYWORDS):
            return "risk_sentiment"
        return "other"

    def fetch_classified(self) -> list[dict]:
        items = self.fetch_all()
        for item in items:
            item["event_type"] = self.classify_event(
                item.get("title", ""),
                item.get("summary", ""),
            )
        return items
