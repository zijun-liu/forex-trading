# USD/JPY Forex Financial Advisor Agent

A deterministic quant engine with LLM reasoning for JPY/USD forex trading analysis. The system computes all numeric features algorithmically (technical indicators, macro spreads, risk assessment, regime classification) and uses LLM for exactly 3 reasoning steps: macro interpretation, news analysis, and strategy synthesis.

## Architecture

```
Data Layer (yfinance, FRED [US + Japan macro], RSS, CFTC)
        |
Feature Engine (deterministic)
  - Technical: SMA, MACD, RSI, ATR, Bollinger, S/R
  - Macro: yield spread, real rate spread, carry signal, DXY, oil-JPY
  - Risk: intervention detector, position sizing, volatility regime
  - Regime: carry_trade / risk_off / policy_divergence / intervention / normal
        |
Z-Score Normalization (historical stats)
        |
LLM Reasoning (3 calls)
  1. Macro Interpreter -> bias + reasoning
  2. News Analyst -> bias + events
  3. Strategy Synthesizer -> direction + conviction + levels
        |
Confidence Calibration (deterministic: penalizes signal disagreement)
        |
Advisory Report
```

## Quick Start

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
# Edit .env with your OPENAI_API_KEY and FRED_API_KEY

# 3. Run analysis
python -m src.main

# JSON output
python -m src.main --json

# Monitor mode (refresh every 4 hours)
python -m src.main --monitor
```

## Backtesting

The backtest engine runs the deterministic signal path only (no LLM calls), making it fast and reproducible.

```bash
# Run backtest on 5 years of data
python -m src.backtesting.simulator

# Custom period and holding period
python -m src.backtesting.simulator --period 2y --hold-days 10

# Custom spread cost and stop-loss distance
python -m src.backtesting.simulator --spread-pips 3.0 --stop-atr 1.5

# JSON output
python -m src.backtesting.simulator --json
```

Metrics computed: Sharpe ratio, max drawdown, win rate, profit factor, expectancy, stop-hit rate, turnover, performance by regime.

Backtest parameters:
- `--hold-days` (default: 6): maximum days to hold a trade before time-based exit
- `--spread-pips` (default: 2.0): broker spread cost deducted from every trade's PnL
- `--stop-atr` (default: 2.0): stop-loss placed this many ATRs away from entry price

## API Keys Required

| Key | Required | Source |
|-----|----------|--------|
| `OPENAI_API_KEY` | Yes | [OpenAI](https://platform.openai.com/api-keys) |
| `FRED_API_KEY` | Yes | [FRED](https://fred.stlouisfed.org/docs/api/api_key.html) (free) |
| `ALPHA_VANTAGE_API_KEY` | No | [Alpha Vantage](https://www.alphavantage.co/support/#api-key) (free, fallback) |
| `EODHD_API_KEY` | No | [EODHD](https://eodhd.com/) (free tier, economic calendar) |

## Project Structure

```
src/
  main.py              # CLI entry point
  pipeline.py           # Orchestrates the full analysis pipeline
  data/                 # Data fetching with caching + fallbacks
  features/             # Deterministic feature computation
  llm/                  # 3 LLM reasoning steps
  memory/               # Short-term (SQLite) + historical (parquet)
  models/               # Pydantic data contracts
  backtesting/          # Deterministic signal replay + metrics
  utils/                # Config, logging, caching
```

## Key Design Decisions

- **LLM boundary**: LLM never computes numbers. All indicators, spreads, and risk metrics are algorithmic. LLM only interprets and synthesizes.
- **Confidence calibration**: `conviction = |mean(biases)| * (1 - variance(biases)) * 100`. The LLM synthesizer cannot exceed the deterministic score by more than 15 points.
- **Backtesting without LLM**: Deterministic signals are replayed over historical data in seconds. LLM only explains signals in live mode.
- **Two-layer memory**: 90-day SQLite for temporal context + parquet datasets for normalization and backtesting.
