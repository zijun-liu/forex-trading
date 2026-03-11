from __future__ import annotations

import pandas as pd
from fredapi import Fred

from src.utils.cache import cached_get
from src.utils.config import get_env
from src.utils.logger import get_logger

logger = get_logger(__name__)

MACRO_SERIES = {
    "fed_funds_rate": "FEDFUNDS",
    "us_cpi_yoy": "CPIAUCSL",
    "us_10y_yield": "DGS10",
    "us_2y_yield": "DGS2",
}
MACRO_CACHE_TTL = 12 * 3600


class FredClient:
    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or get_env("FRED_API_KEY")
        if not key:
            logger.warning("FRED API key not set; calls may fail")
        self._fred = Fred(api_key=key or "")

    def get_series(
        self, series_id: str, start: str | None = None, end: str | None = None
    ) -> pd.Series:
        kwargs: dict[str, str] = {}
        if start is not None:
            kwargs["observation_start"] = start
        if end is not None:
            kwargs["observation_end"] = end
        try:
            return self._fred.get_series(series_id, **kwargs)
        except (ValueError, Exception) as e:
            logger.exception("fredapi get_series failed", series_id=series_id, error=str(e))
            raise

    def get_latest(self, series_id: str) -> float:
        try:
            s = self._fred.get_series(series_id).dropna()
            if s.empty:
                raise ValueError(f"Empty series for {series_id}")
            return float(s.iloc[-1])
        except (ValueError, Exception) as e:
            logger.exception("fredapi get_latest failed", series_id=series_id, error=str(e))
            raise

    def get_macro_snapshot(self) -> dict[str, float]:
        def _fetch() -> dict[str, float]:
            result: dict[str, float] = {}
            for key, sid in MACRO_SERIES.items():
                if key == "us_cpi_yoy":
                    cpi = self._fred.get_series(sid)
                    if len(cpi) < 13:
                        raise ValueError(f"CPIAUCSL has insufficient data for YoY: {len(cpi)} points")
                    current = float(cpi.iloc[-1])
                    prior_year = float(cpi.iloc[-13])
                    result[key] = (current / prior_year - 1.0) * 100.0
                else:
                    s = self._fred.get_series(sid).dropna()
                    if s.empty:
                        raise ValueError(f"Empty series for {sid}")
                    result[key] = float(s.iloc[-1])
            return result

        try:
            return cached_get("fred:macro_snapshot", _fetch, ttl_seconds=MACRO_CACHE_TTL)
        except (ValueError, Exception) as e:
            logger.exception("fredapi get_macro_snapshot failed", error=str(e))
            raise
