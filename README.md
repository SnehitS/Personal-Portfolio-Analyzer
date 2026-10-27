# Personal-Portfolio-Analyzer

Basic project framework (no code yet) for:
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

> All directories currently contain placeholders only (`.gitkeep`) so the framework is tracked in Git.
