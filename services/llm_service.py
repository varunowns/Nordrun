"""
LLM Service
-----------
Thin wrapper around the Anthropic API. Plugins call `ask()` rather than
touching the API directly — this is the seam where multi-provider support
(GPT, Gemini, local models) gets added later.

Retry policy (Task 2 — Phase 0 hardening):
  Transient API errors (rate limits, 5xx, network blips) are retried with
  full-jitter exponential backoff. The number of attempts and base delay
  are configurable via NORDRUN_LLM_MAX_RETRIES and NORDRUN_LLM_RETRY_DELAY
  environment variables so they can be tuned per environment without code
  changes.

  Non-retryable errors (4xx auth/validation, missing key) are raised
  immediately — retrying them would just waste quota.

Requires: pip install anthropic
"""

import logging
import os
import random
import time

from anthropic import Anthropic, APIConnectionError, APIStatusError, RateLimitError

from config import ANTHROPIC_API_KEY, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from core.plugin_registry import require

log = logging.getLogger(__name__)

_client: Anthropic | None = None

# Retry configuration — overridable via environment variables.
# Defaults: 3 attempts, 1-second base delay, 60-second ceiling.
_MAX_RETRIES = int(os.environ.get("NORDRUN_LLM_MAX_RETRIES", "3"))
_BASE_DELAY = float(os.environ.get("NORDRUN_LLM_RETRY_DELAY", "1.0"))
_MAX_DELAY = 60.0

# HTTP status codes that are worth retrying (server-side transient errors).
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        api_key = LLM_API_KEY or ANTHROPIC_API_KEY
        if not api_key:
            raise RuntimeError(
                "No API key configured. Set ANTHROPIC_API_KEY or "
                "NORDRUN_LLM_API_KEY in your .env file."
            )
        kwargs: dict = {"api_key": api_key}
        if LLM_BASE_URL:
            kwargs["base_url"] = LLM_BASE_URL
        _client = Anthropic(**kwargs)
        log.debug("Anthropic client initialised (model=%s)", LLM_MODEL)
    return _client


def _jitter_delay(attempt: int) -> float:
    """Full-jitter exponential backoff: random in [0, min(cap, base * 2^attempt)]."""
    ceiling = min(_MAX_DELAY, _BASE_DELAY * (2 ** attempt))
    return random.uniform(0, ceiling)


@require("llm:call")
def ask(prompt: str, system: str | None = None, max_tokens: int = 1024) -> str:
    """Send a single-turn prompt to Claude and return the text response.

    Retries up to _MAX_RETRIES times on transient errors (rate limits,
    server errors, network failures) using full-jitter exponential backoff.
    Non-retryable errors (auth failures, bad requests) are raised immediately.
    """
    client = _get_client()
    last_exc: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        try:
            log.debug("LLM call attempt %d/%d (model=%s, max_tokens=%d)",
                      attempt + 1, _MAX_RETRIES, LLM_MODEL, max_tokens)
            message = client.messages.create(
                model=LLM_MODEL,
                max_tokens=max_tokens,
                system=system or "",
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(
                block.text for block in message.content if block.type == "text"
            )
            log.debug("LLM call succeeded on attempt %d", attempt + 1)
            return text

        except RateLimitError as exc:
            last_exc = exc
            delay = _jitter_delay(attempt)
            log.warning(
                "LLM rate-limited (attempt %d/%d). Retrying in %.1fs…",
                attempt + 1, _MAX_RETRIES, delay,
            )
            time.sleep(delay)

        except APIStatusError as exc:
            if exc.status_code not in _RETRYABLE_STATUS:
                # 4xx non-rate-limit: auth failure, bad request — don't retry
                log.error("LLM API error %d (not retryable): %s", exc.status_code, exc)
                raise
            last_exc = exc
            delay = _jitter_delay(attempt)
            log.warning(
                "LLM API error %d (attempt %d/%d). Retrying in %.1fs…",
                exc.status_code, attempt + 1, _MAX_RETRIES, delay,
            )
            time.sleep(delay)

        except APIConnectionError as exc:
            last_exc = exc
            delay = _jitter_delay(attempt)
            log.warning(
                "LLM connection error (attempt %d/%d). Retrying in %.1fs…",
                attempt + 1, _MAX_RETRIES, delay,
            )
            time.sleep(delay)

    log.error("LLM call failed after %d attempts: %s", _MAX_RETRIES, last_exc)
    raise RuntimeError(
        f"LLM call failed after {_MAX_RETRIES} attempts"
    ) from last_exc
