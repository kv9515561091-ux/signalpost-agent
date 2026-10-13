"""
Optional Claude client.

The agent's correctness never depends on this module. Everything it
produces is either verifiable against a source the registry already
confirmed, or omitted. That property is what allows the LLM to be wired
in without making the agent's guarantees conditional on a network call
to a paid API succeeding.

Consequences of that policy, enforced here:

* No credentials is a normal state, not an error. Every entry point has
  a caller-supplied fallback.
* The SDK is imported lazily so the package runs without it installed.
* A refusal or a truncated response is reported, never silently parsed
  into whatever partial JSON happened to arrive.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

MODEL = "claude-opus-5"

DEFAULT_MAX_TOKENS = 8_000
DEFAULT_EFFORT = "high"

# Every JSON schema object sent to the API must set this to false.
SCHEMA_ENVELOPE_KEY = "additionalProperties"


class LLMUnavailable(RuntimeError):
    """The SDK is not installed, or no credentials are configured."""


class LLMRefused(RuntimeError):
    """The model declined the request."""


class LLMTruncated(RuntimeError):
    """The response hit max_tokens and may be incomplete JSON."""


class LLMCallFailed(RuntimeError):
    """Transport, authentication or rate-limit failure."""


@dataclass
class LLMResult:
    """A successful completion."""

    data: Any
    model: str
    stop_reason: str
    input_tokens: int | None = None
    output_tokens: int | None = None


def _sdk():
    """Import the Anthropic SDK lazily, or raise :class:`LLMUnavailable`."""
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise LLMUnavailable(
            "The 'anthropic' package is not installed. "
            "Install it with: pip install -r requirements-llm.txt"
        ) from exc
    return anthropic


def is_available() -> bool:
    """Whether a real Claude call could be made right now."""
    try:
        _sdk()
    except LLMUnavailable:
        return False

    # The SDK also resolves credentials from an on-disk profile, so an
    # unset env var is not conclusive on its own. Constructing the client
    # is the only reliable check short of making a request.
    try:
        _sdk().Anthropic()
        return True
    except Exception:  # noqa: BLE001 - any failure means "not usable"
        return False


def unavailable_reason() -> str | None:
    """Why the LLM is unusable, or ``None`` if it is usable."""
    try:
        _sdk()
    except LLMUnavailable as exc:
        return str(exc)

    if not (
        os.getenv("ANTHROPIC_API_KEY")
        or os.getenv("ANTHROPIC_AUTH_TOKEN")
        or os.getenv("CLAUDE_CODE_OAUTH_TOKEN")
    ):
        # No env credential and no SDK-constructible default client.
        try:
            _sdk().Anthropic()
        except Exception:  # noqa: BLE001
            return (
                "No Anthropic credentials found. Set ANTHROPIC_API_KEY "
                "to enable LLM extraction and explanation."
            )
    return None


def _client():
    return _sdk().Anthropic()


def _text_of(response: Any) -> str:
    return next(
        (block.text for block in response.content if block.type == "text"),
        "",
    )


def _usage(response: Any) -> tuple[int | None, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None
    return (
        getattr(usage, "input_tokens", None),
        getattr(usage, "output_tokens", None),
    )


def complete_json(
    schema: dict[str, Any],
    *,
    system: str,
    user: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    effort: str = DEFAULT_EFFORT,
) -> LLMResult:
    """Ask Claude for JSON conforming to ``schema``.

    Uses ``output_config.format``, which constrains generation to the
    schema rather than merely asking for JSON and hoping.

    Raises:
        LLMUnavailable: No SDK or credentials.
        LLMRefused: ``stop_reason`` was ``refusal``.
        LLMTruncated: ``stop_reason`` was ``max_tokens``.
        LLMCallFailed: Transport, auth or rate-limit failure, or a
            response that was not valid JSON.
    """
    client = _client()

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            system=system,
            thinking={"type": "adaptive"},
            output_config={
                "format": {"type": "json_schema", "schema": schema},
                "effort": effort,
            },
            messages=[{"role": "user", "content": user}],
        )
    except Exception as exc:  # noqa: BLE001 - normalise SDK errors
        raise LLMCallFailed(f"{type(exc).__name__}: {exc}") from exc

    stop_reason = getattr(response, "stop_reason", None)

    if stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        category = getattr(details, "category", None)
        raise LLMRefused(f"model declined the request ({category})")

    if stop_reason == "max_tokens":
        raise LLMTruncated(
            "response hit max_tokens; increase the limit or shorten input"
        )

    raw = _text_of(response)
    if not raw:
        raise LLMCallFailed("model returned no text content")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMCallFailed(f"response was not valid JSON: {exc}") from exc

    input_tokens, output_tokens = _usage(response)

    return LLMResult(
        data=data,
        model=MODEL,
        stop_reason=stop_reason or "unknown",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def complete_text(
    *,
    system: str,
    user: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    effort: str = DEFAULT_EFFORT,
) -> str:
    """Ask Claude for prose, with no schema constraint.

    Raises the same exceptions as :func:`complete_json`.
    """
    client = _client()

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            system=system,
            thinking={"type": "adaptive"},
            output_config={"effort": effort},
            messages=[{"role": "user", "content": user}],
        )
    except Exception as exc:  # noqa: BLE001
        raise LLMCallFailed(f"{type(exc).__name__}: {exc}") from exc

    stop_reason = getattr(response, "stop_reason", None)
    if stop_reason == "refusal":
        raise LLMRefused("model declined the request")
    if stop_reason == "max_tokens":
        raise LLMTruncated("response hit max_tokens")

    text = _text_of(response)
    if not text:
        raise LLMCallFailed("model returned no text content")
    return text