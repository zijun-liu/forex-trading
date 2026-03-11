from __future__ import annotations

import pandas as pd
import yfinance as yf

from src.utils.cache import cached_get
from src.utils.logger import get_logger

logger = get_logger(__name__)

INTRADAY_TTL = 3600
DAILY_TTL = 21600

YIELD_SYMBOLS = {
    "us_10y": "^TNX",
    "us_2y": "2YY=F",
    "vix": "^VIX",
    "dxy": "DX-Y.NYB",
    "oil": "CL=F",
}


def _is_daily_interval(interval: str) -> bool:
    return interval in ("1d", "5d", "1wk", "1mo", "3mo")


class MarketDataProvider:
    def __init__(self) -> None:
        self._log = logger

    def get_fx_history(
        self,
        symbol: str = "JPY=X",
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        ttl = DAILY_TTL if _is_daily_interval(interval) else INTRADAY_TTL
        key = f"fx_history:{symbol}:{period}:{interval}"

        def _fetch() -> pd.DataFrame:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval=interval)
                if df.empty:
                    self._log.warning("empty_history", symbol=symbol, period=period, interval=interval)
                    return pd.DataFrame()
                return df
            except Exception as e:
                self._log.exception("yfinance_fetch_error", symbol=symbol, error=str(e))
                return pd.DataFrame()

        return cached_get(key, _fetch, ttl_seconds=ttl)

    def get_current_price(self, symbol: str = "JPY=X") -> float:
        key = f"current_price:{symbol}"

        def _fetch() -> float:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d", interval="1d")
                if hist.empty:
                    self._log.warning("no_price_data", symbol=symbol)
                    return 0.0
                return float(hist["Close"].iloc[-1])
            except Exception as e:
                self._log.exception("yfinance_price_error", symbol=symbol, error=str(e))
                return 0.0

        return cached_get(key, _fetch, ttl_seconds=INTRADAY_TTL)

    def get_yield_data(self) -> dict[str, float]:
        key = "yield_data:all"

        def _fetch() -> dict[str, float]:
            result: dict[str, float] = {}
            for name, sym in YIELD_SYMBOLS.items():
                try:
                    ticker = yf.Ticker(sym)
                    hist = ticker.history(period="5d", interval="1d")
                    if hist.empty:
                        self._log.warning("no_yield_data", name=name, symbol=sym)
                        result[name] = 0.0
                    else:
                        result[name] = float(hist["Close"].iloc[-1])
                except Exception as e:
                    self._log.exception("yfinance_yield_error", name=name, symbol=sym, error=str(e))
                    result[name] = 0.0
            return result

        return cached_get(key, _fetch, ttl_seconds=INTRADAY_TTL)

    def get_multi_history(
        self,
        symbols: list[str],
        period: str = "1y",
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        ttl = DAILY_TTL if _is_daily_interval(interval) else INTRADAY_TTL
        sym_str = ",".join(sorted(symbols))
        key = f"multi_history:{sym_str}:{period}:{interval}"

        def _fetch() -> dict[str, pd.DataFrame]:
            result: dict[str, pd.DataFrame] = {}
            try:
                data = yf.download(
                    symbols,
                    period=period,
                    interval=interval,
                    group_by="ticker",
                    progress=False,
                    auto_adjust=True,
                )
                if data.empty:
                    self._log.warning("empty_multi_history", symbols=symbols)
                    return {s: pd.DataFrame() for s in symbols}
                if len(symbols) == 1:
                    result[symbols[0]] = data.copy()
                else:
                    for sym in symbols:
                        try:
                            if sym in data.columns.get_level_values(0):
                                result[sym] = data[sym].copy()
                            else:
                                result[sym] = pd.DataFrame()
                        except (KeyError, TypeError):
                            result[sym] = pd.DataFrame()
                return result
            except Exception as e:
                self._log.exception("yfinance_multi_error", symbols=symbols, error=str(e))
                return {s: pd.DataFrame() for s in symbols}

        return cached_get(key, _fetch, ttl_seconds=ttl)
