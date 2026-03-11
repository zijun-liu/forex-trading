from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

_DATASETS_DIR = Path(__file__).resolve().parents[2] / "datasets"


class HistoricalData:
    def __init__(self, datasets_dir: Path | None = None) -> None:
        self._dir = datasets_dir or _DATASETS_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def price_path(self) -> Path:
        return self._dir / "price_history.parquet"

    @property
    def macro_path(self) -> Path:
        return self._dir / "macro_history.parquet"

    def save_price_history(self, df: pd.DataFrame) -> None:
        df.to_parquet(self.price_path, index=True)
        logger.info("saved_price_history", rows=len(df), path=str(self.price_path))

    def save_macro_history(self, df: pd.DataFrame) -> None:
        df.to_parquet(self.macro_path, index=True)
        logger.info("saved_macro_history", rows=len(df), path=str(self.macro_path))

    def load_price_history(self) -> Optional[pd.DataFrame]:
        if not self.price_path.exists():
            logger.warning("no_price_history_file", path=str(self.price_path))
            return None
        return pd.read_parquet(self.price_path)

    def load_macro_history(self) -> Optional[pd.DataFrame]:
        if not self.macro_path.exists():
            logger.warning("no_macro_history_file", path=str(self.macro_path))
            return None
        return pd.read_parquet(self.macro_path)

    def compute_feature_stats(self, df: pd.DataFrame) -> dict[str, dict[str, float]]:
        """Compute mean/std for each numeric column for z-score normalization."""
        stats: dict[str, dict[str, float]] = {}
        for col in df.select_dtypes(include="number").columns:
            clean = df[col].dropna()
            if len(clean) >= 2:
                stats[col] = {
                    "mean": float(clean.mean()),
                    "std": float(clean.std()),
                }
        return stats

    def append_price_row(self, row: pd.DataFrame) -> None:
        existing = self.load_price_history()
        if existing is not None:
            combined = pd.concat([existing, row]).drop_duplicates()
            self.save_price_history(combined)
        else:
            self.save_price_history(row)
