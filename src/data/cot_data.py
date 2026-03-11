from __future__ import annotations

import csv
import io
from datetime import datetime, timedelta

import requests

from src.utils.cache import cached_get
from src.utils.logger import get_logger

logger = get_logger(__name__)

COT_URL = "https://www.cftc.gov/dea/newcot/deafut.txt"
JPY_CONTRACT_CODE = "099741"
COT_TTL = 86400

_NONCOMMERCIAL_LONG_PATTERNS = ("noncommercial_long", "noncommercial_positions_long")
_NONCOMMERCIAL_SHORT_PATTERNS = ("noncommercial_short", "noncommercial_positions_short")
_DATE_PATTERNS = ("as_of_date", "report_date", "asofdate")


def _find_column_index(headers: list[str], patterns: tuple[str, ...]) -> int | None:
    for i, h in enumerate(headers):
        hlo = h.lower().replace(" ", "_").replace("-", "_")
        if any(p in hlo for p in patterns):
            return i
    return None


def _parse_int(val: str) -> int:
    try:
        return int(val.replace(",", "").strip())
    except (ValueError, AttributeError):
        return 0


class CotDataClient:
    def __init__(self) -> None:
        self._log = logger

    def fetch_latest(self) -> dict | None:
        def _fetch() -> dict | None:
            try:
                resp = requests.get(
                    COT_URL,
                    timeout=30,
                    headers={"User-Agent": "Mozilla/5.0 (compatible; ForexTrading/1.0)"},
                )
                resp.raise_for_status()
            except requests.RequestException as e:
                self._log.warning("cot_fetch_error", url=COT_URL, error=str(e))
                return None

            reader = csv.reader(io.StringIO(resp.text))
            rows = list(reader)
            if len(rows) < 2:
                return None

            headers = [h.strip() for h in rows[0]]
            long_idx = _find_column_index(headers, _NONCOMMERCIAL_LONG_PATTERNS)
            short_idx = _find_column_index(headers, _NONCOMMERCIAL_SHORT_PATTERNS)
            date_idx = _find_column_index(headers, _DATE_PATTERNS)
            market_idx = next(
                (i for i, h in enumerate(headers) if "market" in h.lower() and "exchange" in h.lower()),
                None,
            )
            code_idx = next(
                (i for i, h in enumerate(headers) if "contract_market_code" in h.lower() or "cfdc" in h.lower()),
                None,
            )

            if long_idx is None or short_idx is None:
                self._log.warning("cot_column_not_found", headers=headers[:10])
                return None

            for row in rows[1:]:
                if len(row) <= max(long_idx, short_idx):
                    continue
                market = row[market_idx] if market_idx is not None and market_idx < len(row) else ""
                code = row[code_idx] if code_idx is not None and code_idx < len(row) else ""
                if "JAPANESE YEN" in market.upper() or JPY_CONTRACT_CODE in code:
                    long_pos = _parse_int(row[long_idx])
                    short_pos = _parse_int(row[short_idx])
                    report_date = ""
                    if date_idx is not None and date_idx < len(row):
                        raw = row[date_idx].strip().replace("-", "").replace("/", "")
                        report_date = row[date_idx].strip()
                        try:
                            if len(raw) >= 8:
                                dt = datetime.strptime(raw[:8], "%Y%m%d")
                                report_date = dt.strftime("%Y-%m-%d")
                            elif len(raw) >= 6:
                                dt = datetime.strptime(raw[:6], "%y%m%d")
                                report_date = dt.strftime("%Y-%m-%d")
                        except ValueError:
                            pass

                    return {
                        "net_positioning": long_pos - short_pos,
                        "net_change": 0,
                        "date": report_date,
                    }

            return None

        return cached_get("cot_jpy_latest", _fetch, ttl_seconds=COT_TTL)

    def get_positioning_delta(self, weeks: int = 4) -> float | None:
        latest = self.fetch_latest()
        if latest is None:
            return None
        current_net = latest.get("net_positioning", 0)
        target_date = datetime.now() - timedelta(weeks=weeks)
        key = f"cot_jpy_history:{target_date.year}:{weeks}"

        def _hist_fetch() -> dict | None:
            url = f"https://www.cftc.gov/files/dea/history/deafut{target_date.year}.txt"
            try:
                r = requests.get(
                    url,
                    timeout=30,
                    headers={"User-Agent": "Mozilla/5.0 (compatible; ForexTrading/1.0)"},
                )
                r.raise_for_status()
            except requests.RequestException:
                return None
            reader = csv.reader(io.StringIO(r.text))
            rows = list(reader)
            if len(rows) < 2:
                return None
            headers = [h.strip() for h in rows[0]]
            long_idx = _find_column_index(headers, _NONCOMMERCIAL_LONG_PATTERNS)
            short_idx = _find_column_index(headers, _NONCOMMERCIAL_SHORT_PATTERNS)
            date_idx = _find_column_index(headers, _DATE_PATTERNS)
            market_idx = next((i for i, h in enumerate(headers) if "market" in h.lower() and "exchange" in h.lower()), None)
            code_idx = next((i for i, h in enumerate(headers) if "contract_market_code" in h.lower() or "cfdc" in h.lower()), None)
            if long_idx is None or short_idx is None:
                return None
            best = None
            best_diff: float = float("inf")
            for row in rows[1:]:
                if len(row) <= max(long_idx, short_idx):
                    continue
                market = row[market_idx] if market_idx is not None and market_idx < len(row) else ""
                code = row[code_idx] if code_idx is not None and code_idx < len(row) else ""
                if "JAPANESE YEN" not in market.upper() and JPY_CONTRACT_CODE not in code:
                    continue
                long_pos = _parse_int(row[long_idx])
                short_pos = _parse_int(row[short_idx])
                net = long_pos - short_pos
                if date_idx is not None and date_idx < len(row):
                    raw = row[date_idx].strip().replace("-", "").replace("/", "")
                    try:
                        row_date = datetime.strptime(raw[:8], "%Y%m%d") if len(raw) >= 8 else datetime.strptime(raw[:6], "%y%m%d")
                    except ValueError:
                        continue
                    diff = abs((row_date - target_date).total_seconds())
                    if diff < best_diff:
                        best_diff = diff
                        best = net
                else:
                    best = net
                    break
            return {"net_positioning": best} if best is not None else None

        hist = cached_get(key, _hist_fetch, ttl_seconds=COT_TTL)
        if hist is None:
            return None
        prev_net = hist.get("net_positioning")
        if prev_net is None:
            return None
        return float(current_net - prev_net)
