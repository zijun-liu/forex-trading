from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd

from src.data.cot_data import CotDataClient
from src.data.economic_calendar import EconomicCalendar
from src.data.fred_client import FredClient
from src.data.market_data import MarketDataProvider
from src.data.news_fetcher import NewsFetcher
from src.features.macro import compute_macro
from src.features.normalize import compute_z_scores
from src.features.regime import classify_regime, detect_rate_change
from src.features.risk import compute_risk
from src.features.technical import compute_technical
from src.llm.client import LLMClient
from src.llm.macro_interpreter import interpret_macro
from src.llm.news_analyst import analyze_news
from src.llm.strategy_synthesizer import calibrate_confidence, synthesize_strategy
from src.memory.historical import HistoricalData
from src.memory.short_term import ShortTermMemory
from src.models.features import FeatureVector
from src.models.report import AdvisoryReport, FXYSnapshot
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ForexAdvisorPipeline:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.market = MarketDataProvider()
        self.fred = FredClient()
        self.news = NewsFetcher()
        self.calendar = EconomicCalendar()
        self.cot = CotDataClient()
        self.memory = ShortTermMemory()
        self.historical = HistoricalData()
        self.llm = LLMClient()

    def run(self) -> AdvisoryReport:
        logger.info("pipeline_start")

        # ── Step 1: Fetch data ────────────────────────────────────────
        logger.info("step_1_fetch_data")
        fx_history = self.market.get_fx_history("JPY=X", period="1y", interval="1d")
        current_price = float(fx_history["Close"].iloc[-1]) if not fx_history.empty else 0.0
        yields = self.market.get_yield_data()

        macro_snapshot = self._safe_macro_snapshot()
        classified_news = self.news.fetch_classified()
        calendar_events = self._safe_calendar()
        cot = self.cot.fetch_latest()
        cot_delta = self.cot.get_positioning_delta(weeks=4)

        # ── Step 2: Compute deterministic features ────────────────────
        logger.info("step_2_features")
        tech_signal = compute_technical(fx_history, current_price)

        dxy_hist = self._get_dxy_history()
        oil_usdjpy = self._get_oil_usdjpy_history(fx_history)

        macro_features = compute_macro(
            us_10y=yields.get("us_10y"),
            us_2y=yields.get("us_2y"),
            jgb_10y=macro_snapshot.get("jgb_10y"),
            us_cpi=macro_snapshot.get("us_cpi_yoy"),
            jp_cpi=macro_snapshot.get("jp_cpi_yoy"),
            us_rate=macro_snapshot.get("fed_funds_rate"),
            jp_rate=macro_snapshot.get("jp_rate"),
            dxy_history=dxy_hist,
            oil_usdjpy_history=oil_usdjpy,
        )

        risk_assessment = compute_risk(
            current_price=current_price,
            atr=tech_signal.atr,
            price_history=fx_history["Close"],
            settings=self.settings,
            cot_net=cot["net_positioning"] if cot else None,
        )

        usdjpy_5d_change = 0.0
        if len(fx_history) >= 6:
            usdjpy_5d_change = (
                fx_history["Close"].iloc[-1] / fx_history["Close"].iloc[-6] - 1
            ) * 100

        rate_changed = self._detect_rate_changes()

        regime = classify_regime(
            vix=yields.get("vix"),
            usdjpy_price=current_price,
            usdjpy_5d_change=usdjpy_5d_change,
            yield_spread=macro_features.yield_spread,
            vol_percentile=risk_assessment.volatility_percentile,
            intervention_risk=risk_assessment.intervention_risk,
            rate_changed_recently=rate_changed,
            settings=self.settings,
        )

        # ── Step 3: Normalize features ────────────────────────────────
        feature_vector = FeatureVector(
            technical=tech_signal,
            macro=macro_features,
            risk=risk_assessment,
            regime=regime,
        )

        hist_stats = self._get_historical_stats()
        normalized = compute_z_scores(feature_vector, hist_stats)
        feature_vector.normalized = normalized

        # ── Step 4: LLM reasoning (3 calls) ───────────────────────────
        logger.info("step_4_llm_reasoning")
        trend_context = self._get_trend_context()

        macro_dict = macro_features.model_dump()
        macro_analysis = interpret_macro(
            self.llm, macro_dict, normalized, trend_context
        )

        news_analysis = analyze_news(self.llm, classified_news, calendar_events)

        signal = synthesize_strategy(
            self.llm, feature_vector, macro_analysis, news_analysis
        )

        # ── Step 5: Build report + store in memory ────────────────────
        logger.info("step_5_report")
        pre_conviction = calibrate_confidence(
            tech_signal.bias, macro_analysis.bias, news_analysis.bias
        )

        decision_log = {
            "timestamp": datetime.now().isoformat(),
            "regime": regime.regime.value,
            "tech_bias": tech_signal.bias,
            "macro_bias": macro_analysis.bias,
            "news_bias": news_analysis.bias,
            "pre_conviction": round(pre_conviction, 2),
            "final_conviction": signal.conviction,
            "direction": signal.direction.value,
            "entry": signal.entry_price,
            "stop": signal.stop_loss,
            "target": signal.take_profit,
        }
        logger.info("decision_trace", **decision_log)

        self.memory.store(
            dt=date.today(),
            features=normalized,
            regime=regime.regime.value,
            sentiment_score=news_analysis.bias,
            cot_net=cot["net_positioning"] if cot else None,
            notes=signal.reasoning[:200],
        )

        fxy_snapshot = self._get_fxy_snapshot(signal)

        report = AdvisoryReport(
            timestamp=datetime.now(),
            pair="JPYUSD",
            current_price=current_price,
            regime=regime,
            features=feature_vector,
            macro_analysis=macro_analysis,
            news_analysis=news_analysis,
            signal=signal,
            fxy=fxy_snapshot,
            decision_log=decision_log,
        )

        logger.info("pipeline_complete", direction=signal.direction.value,
                     conviction=signal.conviction)
        return report

    # ── Helpers ────────────────────────────────────────────────────────

    def _safe_macro_snapshot(self) -> dict[str, float]:
        try:
            return self.fred.get_macro_snapshot()
        except Exception:
            logger.warning("fred_unavailable_using_defaults")
            return {}

    def _safe_calendar(self) -> list[dict]:
        try:
            jp = self.calendar.get_jp_events()
            us = self.calendar.get_us_events()
            return jp + us
        except Exception:
            return []

    def _get_dxy_history(self) -> pd.Series | None:
        try:
            dxy_df = self.market.get_fx_history("DX-Y.NYB", period="3mo")
            if not dxy_df.empty:
                return dxy_df["Close"]
        except Exception:
            pass
        return None

    def _get_oil_usdjpy_history(self, fx_df: pd.DataFrame) -> pd.DataFrame | None:
        try:
            oil_df = self.market.get_fx_history("CL=F", period="3mo")
            if oil_df.empty or fx_df.empty:
                return None
            combined = pd.DataFrame({
                "oil": oil_df["Close"],
                "usdjpy": fx_df["Close"],
            }).dropna()
            return combined if not combined.empty else None
        except Exception:
            return None

    def _detect_rate_changes(self) -> bool:
        try:
            fed = self.fred.get_series("FEDFUNDS")
            return detect_rate_change(fed, None, lookback_days=30)
        except Exception:
            return False

    def _get_historical_stats(self) -> dict[str, dict[str, float]]:
        macro_hist = self.historical.load_macro_history()
        if macro_hist is not None:
            return self.historical.compute_feature_stats(macro_hist)
        return {}

    def _get_trend_context(self) -> list[dict]:
        features_to_track = [
            "yield_spread", "carry_signal", "rsi", "tech_bias",
            "dxy_momentum_5d", "real_rate_spread",
        ]
        trends = []
        for feat in features_to_track:
            trend = self.memory.get_trend_summary(feat, days=30)
            if trend.get("data_points", 0) > 0:
                trends.append(trend)
        return trends

    def _get_fxy_snapshot(self, signal) -> FXYSnapshot | None:
        try:
            fxy_df = self.market.get_fx_history("FXY", period="3mo", interval="1d")
            if fxy_df.empty or len(fxy_df) < 2:
                return None

            close = fxy_df["Close"]
            price = float(close.iloc[-1])

            change_1d = float((close.iloc[-1] / close.iloc[-2] - 1) * 100) if len(close) >= 2 else None
            change_5d = float((close.iloc[-1] / close.iloc[-6] - 1) * 100) if len(close) >= 6 else None
            change_20d = float((close.iloc[-1] / close.iloc[-21] - 1) * 100) if len(close) >= 21 else None

            from ta.trend import SMAIndicator
            from ta.momentum import RSIIndicator
            sma_20 = float(SMAIndicator(close, 20).sma_indicator().iloc[-1]) if len(close) >= 20 else None
            sma_50 = float(SMAIndicator(close, 50).sma_indicator().iloc[-1]) if len(close) >= 50 else None
            rsi = float(RSIIndicator(close, 14).rsi().iloc[-1]) if len(close) >= 15 else None

            direction = signal.direction.value
            conviction = signal.conviction
            if direction == "long_jpyusd":
                if conviction >= 50:
                    rec = "BUY FXY -- strong yen-bullish signal"
                elif conviction >= 25:
                    rec = "Consider buying FXY -- moderate yen-bullish signal"
                else:
                    rec = "Weak buy signal for FXY -- low conviction"
            elif direction == "short_jpyusd":
                if conviction >= 50:
                    rec = "AVOID / SELL FXY -- strong yen-bearish signal"
                elif conviction >= 25:
                    rec = "Lean avoid FXY -- moderate yen-bearish signal"
                else:
                    rec = "Slight caution on FXY -- weak bearish signal"
            else:
                rec = "HOLD / WAIT -- no clear directional signal"

            return FXYSnapshot(
                price=round(price, 2),
                change_1d_pct=round(change_1d, 2) if change_1d is not None else None,
                change_5d_pct=round(change_5d, 2) if change_5d is not None else None,
                change_20d_pct=round(change_20d, 2) if change_20d is not None else None,
                sma_20=round(sma_20, 2) if sma_20 is not None else None,
                sma_50=round(sma_50, 2) if sma_50 is not None else None,
                rsi=round(rsi, 1) if rsi is not None else None,
                recommendation=rec,
            )
        except Exception as e:
            logger.warning("fxy_fetch_error", error=str(e))
            return None

    def close(self) -> None:
        self.memory.close()
