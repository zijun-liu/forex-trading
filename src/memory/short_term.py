from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)

_DB_DIR = Path(__file__).resolve().parents[2] / ".data"
_DB_PATH = _DB_DIR / "memory.db"


class ShortTermMemory:
    def __init__(self, db_path: Path | None = None, retention_days: int = 90) -> None:
        self._retention = retention_days
        path = db_path or _DB_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_state (
                dt TEXT PRIMARY KEY,
                features TEXT,
                regime TEXT,
                sentiment_score REAL,
                cot_net REAL,
                notes TEXT
            )
        """)
        self._conn.commit()

    def store(
        self,
        dt: date,
        features: dict | None = None,
        regime: str | None = None,
        sentiment_score: float | None = None,
        cot_net: float | None = None,
        notes: str | None = None,
    ) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO daily_state
               (dt, features, regime, sentiment_score, cot_net, notes)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                dt.isoformat(),
                json.dumps(features) if features else None,
                regime,
                sentiment_score,
                cot_net,
                notes,
            ),
        )
        self._conn.commit()
        self._prune()

    def get_recent(self, days: int | None = None) -> list[dict[str, Any]]:
        n = days or self._retention
        cutoff = (datetime.now() - timedelta(days=n)).date().isoformat()
        rows = self._conn.execute(
            "SELECT dt, features, regime, sentiment_score, cot_net, notes "
            "FROM daily_state WHERE dt >= ? ORDER BY dt",
            (cutoff,),
        ).fetchall()
        result = []
        for row in rows:
            entry: dict[str, Any] = {"date": row[0]}
            if row[1]:
                try:
                    entry["features"] = json.loads(row[1])
                except json.JSONDecodeError:
                    entry["features"] = {}
            entry["regime"] = row[2]
            entry["sentiment_score"] = row[3]
            entry["cot_net"] = row[4]
            entry["notes"] = row[5]
            result.append(entry)
        return result

    def get_trend_summary(self, feature_name: str, days: int = 30) -> dict[str, Any]:
        """Extract a feature trend over N days for LLM context."""
        recent = self.get_recent(days)
        values = []
        for entry in recent:
            feats = entry.get("features", {})
            if isinstance(feats, dict) and feature_name in feats:
                val = feats[feature_name]
                if val is not None:
                    values.append(val)

        if not values:
            return {"feature": feature_name, "data_points": 0}

        return {
            "feature": feature_name,
            "data_points": len(values),
            "latest": values[-1],
            "earliest": values[0],
            "change": values[-1] - values[0],
            "direction": "increasing" if values[-1] > values[0] else "decreasing",
            "min": min(values),
            "max": max(values),
        }

    def get_regime_history(self, days: int = 30) -> list[str]:
        recent = self.get_recent(days)
        return [e.get("regime", "unknown") for e in recent if e.get("regime")]

    def _prune(self) -> None:
        cutoff = (datetime.now() - timedelta(days=self._retention)).date().isoformat()
        self._conn.execute("DELETE FROM daily_state WHERE dt < ?", (cutoff,))
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
