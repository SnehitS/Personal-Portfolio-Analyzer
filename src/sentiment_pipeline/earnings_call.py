"""Summarize an earnings call transcript with Claude: a few bullets a shareholder would want."""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-5-5"

SYSTEM = """You summarize earnings calls for someone who owns the stock.

Write 4 to 6 bullets, each starting with "- " and one or two sentences long. Cover, where \
the call says something about them: results against expectations, guidance and whether it \
was raised or cut, management's tone, the main risks or problems raised, and anything \
analysts pushed back on. Use the call's own numbers. Don't add a heading, an intro or a \
conclusion, and don't give investment advice."""


def transcript_text(entries: list[dict]) -> str:
    return "\n\n".join(
        f"{e.get('speaker', '')}" + (f" ({e['title']})" if e.get("title") else "")
        + f": {e.get('content', '')}" for e in entries)


def summarize(ticker: str, quarter: str, entries: list[dict], model: str = DEFAULT_MODEL,
              client=None) -> str | None:
    """Bullet summary, or None if Claude isn't available or the request failed."""
    try:
        import anthropic
    except ImportError:
        log.info("anthropic not installed; earnings calls are stored but not summarized")
        return None
    from src.sentiment_pipeline.llm import has_credentials
    if client is None:
        if not has_credentials():
            log.info("No Anthropic key; earnings calls are stored but not summarized")
            return None
        client = anthropic.Anthropic()
    try:
        response = client.messages.create(
            model=model,
            max_tokens=2000,
            system=SYSTEM,
            messages=[{"role": "user", "content":
                       f"{ticker} earnings call, {quarter}:\n\n{transcript_text(entries)}"}],
            output_config={"effort": "low"},
        )
    except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
        log.warning("Earnings call summary for %s failed: %s", ticker, exc)
        return None
    if response.stop_reason == "refusal":
        log.warning("Claude declined to summarize the %s call (request %s)", ticker,
                    response._request_id)
        return None
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    bullets = [line.strip() for line in text.splitlines() if line.strip().startswith("- ")]
    return "\n".join(bullets) or text or None
