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
│   ├── news_raw/                # Cached SEC ticker -> CIK map
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

Performance: net deposits, gain and money-weighted (XIRR) return since your first deposit, next to what the same deposits would be worth in each benchmark (SPY, QQQ by default).

Tax lots are rebuilt FIFO from order history, with stock splits applied from yfinance and trade dates in New York time. If a holding's lots don't add up to its share count (ACATS transfers, stock rewards, spinoffs), it's flagged and its lot dates and gains shouldn't be trusted. A snapshot only uses transactions from its own source, so mixing `snapshot` and `snapshot --csv` doesn't double-count.

- Robinhood access uses the unofficial `robin_stocks` library and is **read-only**: only read functions are imported.
- The first login will likely ask you to approve the device in the Robinhood app; the session is cached after that.
- Snapshots go to `data/processed/portfolio.db` (SQLite). `data/`, `reports/`, `.env`, `*.csv` and `*.db` are gitignored.
- Thresholds for risk flags live in `configs/settings.yaml`.

## Stage 2: News, sentiment, catalysts and options (done)

```bash
pip install -r requirements-sentiment.txt   # optional: FinBERT and/or Claude sentiment
python -m src.cli news                # fetch + score news for every holding
python -m src.cli news NVDA --why     # one ticker, with the LLM's reason per headline
python -m src.cli catalysts --moves   # next 30 days of events, with implied earnings moves
python -m src.cli options             # IV, realized vol, expected moves, earnings vs history
```

With no tickers, each command uses the latest snapshot's holdings (and option underlyings).

**News** comes from every provider you have set up, de-duplicated across them:

| Source | Key | What it adds |
|---|---|---|
| Yahoo Finance RSS | none | latest ~20 headlines per ticker |
| SEC EDGAR | none (set `SEC_USER_AGENT`) | 8-K (item codes decoded), 10-Q/10-K, Form 4 insider trades, 13D/G, S-3 |
| Finnhub | `FINNHUB_API_KEY` | company news with summaries |
| Alpha Vantage | `ALPHAVANTAGE_API_KEY` | articles with its own per-ticker sentiment and relevance (25 calls/day, capped per run) |

**Sentiment** gives each article one score from -1 to +1, using the best model available: Claude (`ANTHROPIC_API_KEY`), then local FinBERT, then Alpha Vantage's own score. Filings aren't scored. Claude reads all of a ticker's headlines in one request, judges the effect on that stock rather than the article's tone, and gives a one-line reason. Each article is only scored once per model. A ticker's read is a weighted average: newer articles count more (48h half-life), and so do relevant ones. Headlines that never mention the company, like Yahoo's market roundups, count for 10%.

**Catalysts:** earnings (with before/after-market timing and EPS estimate from Finnhub), ex-dividend dates, expiries of options you hold, tax lots turning long-term, and macro dates you list under `catalysts.macro_events` in `configs/settings.yaml`.

**Options** (from yfinance chains):
- Expected move to an expiry is the at-the-money straddle price.
- 30-day IV is interpolated across expiries, next to 30-day realized volatility.
- The earnings move isolates the event: the post-earnings straddle minus the variance of ordinary days. It's shown next to the stock's average actual move over recent reports.
- Each run stores that day's 30-day IV, so IV rank appears once 20+ days have built up. Run `options` daily, e.g. from Task Scheduler.

News, scores and IV history live in the same SQLite file as the snapshots.

## Roadmap
- **Stage 3:** daily/weekly reports with position moves, buy/sell/trim ideas, cash deployment, new long/short and options ideas, and later prediction markets.
