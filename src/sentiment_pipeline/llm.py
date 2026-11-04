"""LLM sentiment via the Claude API: reads a ticker's headlines together, so it can tell
"Apple beats" from "Apple's supplier beats" and explain each call in a sentence.

Needs ANTHROPIC_API_KEY (or an `ant auth login` profile). One request per ticker.
"""

from __future__ import annotations

import json
import logging

from src.data_pipeline.models import NewsItem, SentimentScore
from src.sentiment_pipeline.finbert import text_for

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5-5"
LABELS = ("positive", "negative", "neutral")

SYSTEM = """You rate financial news for someone who holds the stock.

For each article, judge the likely effect on the given ticker's share price, not the general \
mood of the writing. Market-wide roundups that only mention the ticker in passing, or stories \
about other companies, are neutral unless they clearly affect this one. Score from -1 (clearly \
bad for the stock) to +1 (clearly good); keep scores near 0 when the impact is small or \
uncertain. Confidence is 0 to 1. The rationale is one short sentence naming the reason."""

SCHEMA = {
    "type": "object",
    "properties": {
        "articles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "label": {"type": "string", "enum": list(LABELS)},
                    "score": {"type": "number"},
                    "confidence": {"type": "number"},
                    "rationale": {"type": "string"},
                },
                "required": ["id", "label", "score", "confidence", "rationale"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["articles"],
    "additionalProperties": False,
}


def available() -> bool:
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def build_prompt(ticker: str, name: str | None, items: list[NewsItem]) -> str:
    lines = [f"Ticker: {ticker}" + (f" ({name})" if name else ""), "", "Articles:"]
    for i, n in enumerate(items):
        lines.append(f"[{i}] {n.published_at:%Y-%m-%d} {n.publisher or n.source}: {text_for(n)}")
    return "\n".join(lines)


def parse_response(ticker: str, items: list[NewsItem], text: str,
                   model: str) -> list[SentimentScore]:
    out = []
    for row in json.loads(text).get("articles", []):
        i = row.get("id")
        if not isinstance(i, int) or not 0 <= i < len(items) or row.get("label") not in LABELS:
            continue
        out.append(SentimentScore(
            ticker, items[i].url, model, row["label"],
            max(-1.0, min(1.0, float(row["score"]))),
            max(0.0, min(1.0, float(row["confidence"]))), row.get("rationale", "").strip(),
        ))
    return out


class LlmScorer:
    def __init__(self, model: str = DEFAULT_MODEL, batch_size: int = 25, client=None):
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.batch_size = batch_size

    def score(self, items: list[NewsItem], names: dict[str, str] | None = None
              ) -> list[SentimentScore]:
        names = names or {}
        articles = [n for n in items if n.kind == "article"]
        by_ticker: dict[str, list[NewsItem]] = {}
        for n in articles:
            by_ticker.setdefault(n.ticker, []).append(n)
        out: list[SentimentScore] = []
        for ticker, group in by_ticker.items():
            for start in range(0, len(group), self.batch_size):
                chunk = group[start:start + self.batch_size]
                out += self._score_chunk(ticker, names.get(ticker), chunk)
        return out

    def _score_chunk(self, ticker: str, name: str | None,
                     items: list[NewsItem]) -> list[SentimentScore]:
        import anthropic

        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                system=SYSTEM,
                messages=[{"role": "user", "content": build_prompt(ticker, name, items)}],
                output_config={"effort": "low",
                               "format": {"type": "json_schema", "schema": SCHEMA}},
                # If a safety classifier declines, the API retries on a fallback model
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.AuthenticationError:
            log.warning("Anthropic API key missing or invalid; skipping LLM sentiment")
            raise
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
            log.warning("LLM sentiment for %s failed: %s", ticker, exc)
            return []
        if response.stop_reason == "refusal":
            log.warning("LLM declined to score %s news (request %s)", ticker,
                        response._request_id)
            return []
        if response.stop_reason == "max_tokens":
            log.warning("LLM output for %s was cut off; try a smaller batch_size", ticker)
            return []
        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            return parse_response(ticker, items, text, self.model)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            log.warning("LLM sentiment for %s didn't parse: %s", ticker, exc)
            return []
