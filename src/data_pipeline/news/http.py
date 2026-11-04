"""Small shared HTTP helper: timeouts, a User-Agent, and per-host pacing for rate limits."""

from __future__ import annotations

import logging
import time
from urllib.parse import urlparse

import requests

log = logging.getLogger(__name__)

DEFAULT_UA = "Personal-Portfolio-Analyzer/0.2"
_last_call: dict[str, float] = {}
_session = requests.Session()


def get(url: str, params: dict | None = None, *, min_interval: float = 0.0,
        user_agent: str = DEFAULT_UA, timeout: float = 15.0) -> requests.Response | None:
    """GET with at least `min_interval` seconds between calls to the same host.
    Returns None (and logs) on network errors or non-2xx responses."""
    host = urlparse(url).netloc
    wait = _last_call.get(host, 0.0) + min_interval - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    try:
        resp = _session.get(url, params=params, timeout=timeout,
                            headers={"User-Agent": user_agent})
    except requests.RequestException as exc:
        log.warning("request to %s failed: %s", host, exc)
        return None
    finally:
        _last_call[host] = time.monotonic()
    if not resp.ok:
        log.warning("%s returned HTTP %s", host, resp.status_code)
        return None
    return resp
