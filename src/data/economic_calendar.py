from __future__ import annotations

from datetime import datetime

import requests

from src.utils.cache import cached_get
from src.utils.config import get_env
from src.utils.logger import get_logger

logger = get_logger(__name__)

EODHD_BASE = "https://eodhistoricaldata.com/api/economic-events"
CALENDAR_TTL = 3600


class EconomicCalendar:
    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or get_env("EODHD_API_KEY")

    def get_upcoming_events(
        self,
        country: str = "JP",
        days_ahead: int = 7,
    ) -> list[dict]:
        today = datetime.now().strftime("%Y-%m-%d")
        key = f"econ_calendar:{country}:{today}:{days_ahead}"

        def _fetch() -> list[dict]:
            if not self._api_key:
                logger.warning("EODHD API key not set; skipping calendar")
                return []
            try:
                params = {
                    "api_token": self._api_key,
                    "country": country,
                    "fmt": "json",
                    "from": today,
                }
                resp = requests.get(EODHD_BASE, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                if not isinstance(data, list):
                    return []
                return [
                    {
                        "date": e.get("date", ""),
                        "event": e.get("event", ""),
                        "country": e.get("country", ""),
                        "actual": e.get("actual"),
                        "previous": e.get("previous"),
                        "estimate": e.get("estimate"),
                        "impact": e.get("impact", ""),
                    }
                    for e in data[:50]
                ]
            except Exception as e:
                logger.warning("eodhd_calendar_error", error=str(e))
                return []

        return cached_get(key, _fetch, ttl_seconds=CALENDAR_TTL)

    def get_us_events(self, days_ahead: int = 7) -> list[dict]:
        return self.get_upcoming_events(country="US", days_ahead=days_ahead)

    def get_jp_events(self, days_ahead: int = 7) -> list[dict]:
        return self.get_upcoming_events(country="JP", days_ahead=days_ahead)
