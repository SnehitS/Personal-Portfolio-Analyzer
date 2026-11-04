"""FinBERT (ProsusAI/finbert): a BERT model fine-tuned on financial news, run locally on CPU.

Optional: needs `pip install -r requirements-sentiment.txt`. The model (~440 MB) downloads
from Hugging Face on first use and is cached after that.
"""

from __future__ import annotations

import logging
import os

from src.data_pipeline.models import NewsItem, SentimentScore

log = logging.getLogger(__name__)

MODEL_ID = "ProsusAI/finbert"
MODEL_NAME = "finbert"


def available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def text_for(item: NewsItem) -> str:
    """Headline plus the start of the summary; FinBERT sees at most 512 tokens anyway."""
    summary = item.summary[:400] if item.summary and item.summary != item.headline else ""
    return f"{item.headline}. {summary}".strip()


def to_score(item: NewsItem, probs: dict[str, float]) -> SentimentScore:
    pos, neg = probs.get("positive", 0.0), probs.get("negative", 0.0)
    label = max(probs, key=probs.get)
    return SentimentScore(item.ticker, item.url, MODEL_NAME, label, round(pos - neg, 4),
                          round(probs[label], 4))


class FinBertScorer:
    def __init__(self, batch_size: int = 16):
        # Windows without Developer Mode can't symlink the cache; it still works, just noisily
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        from transformers import pipeline
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        self.batch_size = batch_size
        self._pipe = pipeline("text-classification", model=MODEL_ID, top_k=None,
                              truncation=True, max_length=512, device="cpu")

    def score(self, items: list[NewsItem]) -> list[SentimentScore]:
        articles = [n for n in items if n.kind == "article"]  # filings have no tone to read
        if not articles:
            return []
        outputs = self._pipe([text_for(n) for n in articles], batch_size=self.batch_size)
        return [to_score(n, {o["label"].lower(): o["score"] for o in out})
                for n, out in zip(articles, outputs)]
