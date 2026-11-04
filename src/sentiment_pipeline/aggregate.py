"""Turn per-article scores into one read per ticker, and run whichever models are set up."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.data_pipeline import storage
from src.data_pipeline.models import NewsItem, SentimentScore

log = logging.getLogger(__name__)


@dataclass
class ArticleRead:
    item: NewsItem
    score: float
    label: str
    model: str            # which model the score came from
    confidence: float | None = None
    rationale: str = ""


@dataclass
class TickerSentiment:
    ticker: str
    score: float | None   # recency-weighted, -1..+1
    articles: int
    positive: int
    negative: int
    most_positive: ArticleRead | None = None
    most_negative: ArticleRead | None = None

    @property
    def label(self) -> str:
        if self.score is None:
            return "-"
        return "bullish" if self.score >= 0.15 else "bearish" if self.score <= -0.15 else "mixed"


def _label(score: float) -> str:
    return "positive" if score >= 0.15 else "negative" if score <= -0.15 else "neutral"


def best_reads(news: list[NewsItem], scores: list[SentimentScore]) -> list[ArticleRead]:
    """One score per article: the LLM's if present, else FinBERT's, else the provider's."""
    by_key: dict[tuple[str, str], dict[str, SentimentScore]] = {}
    for s in scores:
        by_key.setdefault((s.ticker, s.url), {})[s.model] = s
    out = []
    for n in news:
        if n.kind != "article":
            continue
        mine = by_key.get((n.ticker, n.url), {})
        llm = next((s for m, s in mine.items() if m != "finbert"), None)
        pick = llm or mine.get("finbert")
        if pick:
            out.append(ArticleRead(n, pick.score, pick.label, pick.model, pick.confidence,
                                   pick.rationale))
        elif n.provider_sentiment is not None:
            out.append(ArticleRead(n, n.provider_sentiment, _label(n.provider_sentiment),
                                   n.source))
    return out


def summarize(reads: list[ArticleRead], ticker: str, half_life_hours: float = 48,
              now: datetime | None = None) -> TickerSentiment:
    """Weighted mean where each article counts by recency, relevance and confidence."""
    now = now or datetime.now(timezone.utc)
    mine = [r for r in reads if r.item.ticker == ticker]
    if not mine:
        return TickerSentiment(ticker, None, 0, 0, 0)
    total = weight_sum = 0.0
    for r in mine:
        age_h = max(0.0, (now - r.item.published_at).total_seconds() / 3600)
        w = 0.5 ** (age_h / half_life_hours)
        w *= r.item.relevance if r.item.relevance is not None else 1.0
        w *= r.confidence if r.confidence is not None else 1.0
        total += w * r.score
        weight_sum += w
    return TickerSentiment(
        ticker=ticker,
        score=total / weight_sum if weight_sum else None,
        articles=len(mine),
        positive=sum(r.label == "positive" for r in mine),
        negative=sum(r.label == "negative" for r in mine),
        most_positive=max((r for r in mine if r.score > 0), key=lambda r: r.score, default=None),
        most_negative=min((r for r in mine if r.score < 0), key=lambda r: r.score, default=None),
    )


def _scorers(cfg: dict) -> list[tuple[str, object]]:
    """(model name, scorer) for each configured model that can run here."""
    out = []
    for name in cfg.get("models", []):
        if name == "finbert":
            from src.sentiment_pipeline import finbert
            if finbert.available():
                out.append((finbert.MODEL_NAME, finbert.FinBertScorer()))
            else:
                log.info("FinBERT not installed; see requirements-sentiment.txt")
        elif name == "llm":
            from src.sentiment_pipeline import llm
            if not llm.available():
                log.info("anthropic not installed; see requirements-sentiment.txt")
            elif not llm.has_credentials():
                log.warning("LLM sentiment skipped: set ANTHROPIC_API_KEY in .env to enable it")
            else:
                try:
                    model = cfg.get("llm_model", llm.DEFAULT_MODEL)
                    out.append((model, llm.LlmScorer(model, cfg.get("llm_batch_size", 25))))
                except Exception as exc:  # no credentials configured
                    log.warning("LLM sentiment unavailable: %s", exc)
    return out


def score_new(news: list[NewsItem], cfg: dict, db_path: Path,
              names: dict[str, str] | None = None) -> dict[str, int]:
    """Score articles each model hasn't seen yet, store the results, and return how many
    each model scored."""
    articles = [n for n in news if n.kind == "article"]
    if not articles:
        return {}
    counts = {}
    for model, scorer in _scorers(cfg):
        done = {(s.ticker, s.url) for s in storage.load_sentiment(
            sorted({n.ticker for n in articles}), model, db_path)}
        todo = [n for n in articles if (n.ticker, n.url) not in done]
        if not todo:
            continue
        try:
            scores = scorer.score(todo, names)
        except Exception as exc:
            log.warning("%s scoring failed: %s", model, exc)
            continue
        storage.save_sentiment(scores, db_path)
        counts[model] = len(scores)
    return counts
