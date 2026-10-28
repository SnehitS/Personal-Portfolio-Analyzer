# Personal-Portfolio-Analyzer

Personal portfolio analyzer for:
- analyzing a personal stock portfolio,
- running sentiment analysis on current stock news,
- generating different update types (daily/weekly/alerts).

## Project Structure

```
Personal-Portfolio-Analyzer/
├── configs/                     # Config files (API keys/env mappings/thresholds later)
├── data/
│   ├── portfolio_raw/           # Raw portfolio inputs (CSV/JSON exports)
│   ├── news_raw/                # Raw news data from APIs/scrapers
│   └── processed/               # Cleaned/feature-ready datasets
├── docs/                        # Product notes, data dictionary, workflow docs
├── models/
│   ├── trained/                 # Saved model artifacts
│   └── registry/                # Model version tracking metadata
├── notebooks/
│   ├── exploration/             # EDA and idea validation notebooks
│   └── experiments/             # Sentiment/model experiment notebooks
├── reports/
│   ├── daily/                   # Daily portfolio/news updates
│   ├── weekly/                  # Weekly summary updates
│   └── alerts/                  # Risk/event-based update outputs
├── src/
│   ├── data_pipeline/           # Ingestion + preprocessing modules (to be added)
│   ├── sentiment_pipeline/      # News sentiment workflow modules (to be added)
│   ├── portfolio_analysis/      # Portfolio analytics modules (to be added)
│   └── update_engine/           # Update/notification logic modules (to be added)
└── tests/                       # Test structure for future code
```

## Stage 1: Portfolio data (done)

```bash
pip install -r requirements.txt
cp .env.example .env        # add your Robinhood login
python -m src.cli snapshot  # login, pull holdings/options/history, enrich, save
python -m src.cli snapshot --csv path/to/robinhood_activity.csv   # no-login fallback
python -m src.cli history   # past snapshots
python -m src.cli position NVDA   # full deep dive on one holding, incl. tax lots
```

Each snapshot captures, per holding:
- **Position:** shares, average cost, value, weight, unrealized P/L, FIFO tax lots
- **Price behavior:** 1W/1M/3M/YTD/1Y returns, volatility, max drawdown, RSI, 50/200-day trend, beta
- **Valuation & quality:** P/E (ttm/fwd), PEG, P/S, EV/EBITDA, margins, growth, free cash flow, debt/equity
- **Street & positioning:** analyst rating and target range, short interest, institutional/insider ownership
- **Events:** next earnings, last EPS surprise, dividend yield, ex-dividend date

Portfolio level: sector mix, portfolio beta and volatility, highly correlated pairs, realized short/long-term gains and dividend income for the year, plus risk flags.

- Robinhood access uses the unofficial `robin_stocks` library and is **read-only**: only read functions are imported.
- The first login will likely ask you to approve the device in the Robinhood app; the session is cached after that.
- Snapshots go to `data/processed/portfolio.db` (SQLite). `data/`, `reports/`, `.env`, `*.csv` and `*.db` are gitignored.
- Thresholds for risk flags live in `configs/settings.yaml`.

## Roadmap
- **Stage 2:** per-ticker news (Finnhub, Alpha Vantage, SEC EDGAR), FinBERT + LLM sentiment, earnings/catalyst calendar, options data (IV, expected moves).
- **Stage 3:** daily/weekly reports with position moves, buy/sell/trim ideas, cash deployment, new long/short and options ideas, and later prediction markets.
