"""
LLM access, provider-agnostic.

The agent's correctness never depends on this module. Everything it
produces is either verifiable against a source the registry already
confirmed, or omitted. That property is what allows an LLM to be wired
in without making the agent's guarantees conditional on a network call
to a paid API succeeding.

Two backends are supported:

``anthropic``           Claude via the official SDK, with
                        ``output_config.format`` constraining generation
                        to the requested schema.
``openai_compatible``   Any Chat Completions endpoint, including NVIDIA
                        NIM. Constrained decoding is not guaranteed
                        there, so the payload is parsed defensively.

Selection is automatic: an Anthropic key wins if both are present, and
``LLM_PROVIDER`` overrides the guess.

Neither backend is trusted to decide what is true. Quoted evidence is
checked against the fetched page by :mod:`src.extract`, which is our
code and runs identically whichever provider produced the text.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

# Load .env here rather than relying on another module having done it.
# The search provider happens to load it at import time, but extraction
# and narration run without --web, so relying on that would make a key
# in .env silently invisible on exactly the paths that need it.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is a runtime dependency
    pass

ANTHROPIC_MODEL = "claude-opus-5"

DEFAULT_MAX_TOKENS = 8_000
DEFAULT_EFFORT = "high"


class LLMUnavailable(RuntimeError):
    """No backend is configured or usable."""


class LLMRefused(RuntimeError):
    """The provider declined or rejected the request."""


class LLMTruncated(RuntimeError):
    """The response hit a token limit and may be incomplete."""


class LLMCallFailed(RuntimeError):
    """Transport, authentication or rate-limit failure."""


@dataclass
class LLMResult:
    """A successful completion.

    Token counts are retained so a caller can report what a paid API
    call cost, which matters once extraction is running in a batch.
    """

    data: Any
    model: str
    provider: str
    finish_reason: str
    input_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None and self.output_tokens is None:
            return None
        return (self.input_tokens or 0) + (self.output_tokens or 0)


def _anthropic_sdk():
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise LLMUnavailable(
            "The 'anthropic' package is not installed. Install it with: "
            "pip install -r requirements-llm.txt, or set NVIDIA_API_KEY "
            "to use an OpenAI-compatible provider instead."
        ) from exc
    return anthropic


# ---------------------------------------------------------------- Anthropic
def _anthropic_available() -> bool:
    try:
        _anthropic_sdk()
    except LLMUnavailable:
        return False
    try:
        _anthropic_sdk().Anthropic()
        return True
    except Exception:  # noqa: BLE001
        return False


def _anthropic_complete_json(
    schema: dict[str, Any],
    *,
    system: str,
    user: str,
    max_tokens: int,
    effort: str,
) -> LLMResult:
    try:
        response = _anthropic_sdk().Anthropic().messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            system=system,
            thinking={"type": "adaptive"},
            output_config={
                "format": {"type": "json_schema", "schema": schema},
                "effort": effort,
            },
            messages=[{"role": "user", "content": user}],
        )
    except Exception as exc:  # noqa: BLE001
        raise LLMCallFailed(f"{type(exc).__name__}: {exc}") from exc

    stop_reason = getattr(response, "stop_reason", None)

    if stop_reason == "refusal":
        raise LLMRefused("model declined the request")
    if stop_reason == "max_tokens":
        raise LLMTruncated("response hit max_tokens; output may be incomplete")

    text = next(
        (b.text for b in response.content if b.type == "text"), ""
    )
    if not text:
        raise LLMCallFailed("model returned no text content")

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMCallFailed(f"response was not valid JSON: {exc}") from exc

    usage = getattr(response, "usage", None)

    return LLMResult(
        data=data,
        model=ANTHROPIC_MODEL,
        provider="anthropic",
        finish_reason=stop_reason or "unknown",
        input_tokens=getattr(usage, "input_tokens", None),
        output_tokens=getattr(usage, "output_tokens", None),
    )


def _anthropic_complete_text(
    *,
    system: str,
    user: str,
    max_tokens: int,
    effort: str,
) -> str:
    try:
        response = _anthropic_sdk().Anthropic().messages.create(
            model=ANTHROPIC_MODEL,
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

    text = next(
        (b.text for b in response.content if b.type == "text"), ""
    )
    if not text:
        raise LLMCallFailed("model returned no text content")
    return text


# ------------------------------------------------------------------ facade
# Credential environment variables, in the order the Anthropic SDK itself
# resolves them.
ANTHROPIC_CREDENTIAL_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
)

OPENAI_CREDENTIAL_VARS = (
    "NVIDIA_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_COMPATIBLE_API_KEY",
)


def _has_env_credential(names: tuple[str, ...]) -> bool:
    return any((os.getenv(name) or "").strip() for name in names)


def _anthropic_configured() -> bool:
    """An Anthropic key was configured explicitly."""
    return _has_env_credential(ANTHROPIC_CREDENTIAL_VARS)


def _openai_configured() -> bool:
    return _has_env_credential(OPENAI_CREDENTIAL_VARS) and (
        _openai_backend() is not None
    )


def _selected_provider() -> str | None:
    """Which backend to use, or ``None`` when none is configured.

    Explicitly configured credentials win over anything merely
    discoverable. The Anthropic SDK resolves credentials from an on-disk
    profile even with no environment variable set, so a developer machine
    with Claude Code installed would otherwise silently use Claude even
    after configuring a different provider.
    """
    override = (os.getenv("LLM_PROVIDER") or "").strip().lower()

    if override in ("anthropic", "claude"):
        return "anthropic" if _anthropic_available() else None
    if override in ("nvidia", "openai", "openai_compatible"):
        return "openai_compatible" if _openai_backend() else None
    if override in ("none", "off", "disabled"):
        return None

    # Configured credentials, Anthropic first since its schema guarantee
    # is the stronger of the two.
    if _anthropic_configured() and _anthropic_available():
        return "anthropic"
    if _openai_configured():
        return "openai_compatible"
    if _anthropic_configured() and _anthropic_available():
        return "anthropic"

    # Nothing configured explicitly. Fall back to a discoverable SDK
    # profile, so a machine authenticated another way still works.
    if _anthropic_available():
        return "anthropic"

    return None


def _openai_backend():
    from src.llm_openai import backend_from_env

    return backend_from_env()


def provider_name() -> str | None:
    """Name of the active backend, or ``None`` if unavailable."""
    return _selected_provider()


def is_available() -> bool:
    """Whether a real model call could be made right now."""
    return _selected_provider() is not None


def unavailable_reason() -> str | None:
    """Why no backend is usable, or ``None`` if one is."""
    if is_available():
        return None

    override = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if override in ("none", "off", "disabled"):
        return "LLM disabled by LLM_PROVIDER"

    if _anthropic_configured() and not _anthropic_available():
        return (
            "ANTHROPIC_API_KEY is set but the anthropic package is not "
            "installed. Run: pip install -r requirements-llm.txt"
        )
    if _openai_configured():
        return "NVIDIA_API_KEY is set but the endpoint rejected a test call"

    if not _anthropic_configured() and not _openai_configured():
        return (
            "No LLM configured. Set ANTHROPIC_API_KEY or NVIDIA_API_KEY "
            "in .env to enable fact extraction and narrative explanation."
        )

    return None


def describe() -> str:
    """One-line description of the active backend, for diagnostics."""
    provider = _selected_provider()
    if provider == "anthropic":
        return f"anthropic:{ANTHROPIC_MODEL}"
    if provider == "openai_compatible":
        backend = _openai_backend()
        return backend.describe() if backend else "openai_compatible"
    return "none"


def complete_json(
    schema: dict[str, Any],
    *,
    system: str,
    user: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    effort: str = DEFAULT_EFFORT,
) -> LLMResult:
    """Return JSON matching ``schema`` from whichever backend is active."""
    provider = _selected_provider()

    if provider == "anthropic":
        return _anthropic_complete_json(
            schema, system=system, user=user,
            max_tokens=max_tokens, effort=effort,
        )

    if provider == "openai_compatible":
        backend = _openai_backend()
        try:
            data = backend.complete_json(
                schema, system=system, user=user, max_tokens=max_tokens
            )
            last_usage = backend.last_usage
        except Exception as exc:  # noqa: BLE001
            raise _translate(exc) from exc

        return LLMResult(
            data=data,
            model=backend.model,
            provider="openai_compatible",
            finish_reason="stop",
            input_tokens=last_usage.get("prompt_tokens"),
            output_tokens=last_usage.get("completion_tokens"),
        )

    raise LLMUnavailable(unavailable_reason() or "no LLM backend configured")


def complete_text(
    *,
    system: str,
    user: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    effort: str = DEFAULT_EFFORT,
) -> str:
    """Return prose from whichever backend is active."""
    provider = _selected_provider()

    if provider == "anthropic":
        return _anthropic_complete_text(
            system=system, user=user, max_tokens=max_tokens, effort=effort
        )

    if provider == "openai_compatible":
        backend = _openai_backend()
        try:
            return backend.complete_text(
                system=system, user=user, max_tokens=max_tokens
            )
        except Exception as exc:  # noqa: BLE001
            raise _translate(exc) from exc

    raise LLMUnavailable(unavailable_reason() or "no LLM backend configured")


def _translate(exc: Exception) -> Exception:
    """Map backend-specific exceptions onto the shared hierarchy."""
    from src.llm_openai import CallFailed, Refused, Truncated

    if isinstance(exc, Refused):
        return LLMRefused(str(exc))
    if isinstance(exc, Truncated):
        return LLMTruncated(str(exc))
    if isinstance(exc, CallFailed):
        return LLMCallFailed(str(exc))
    return LLMCallFailed(f"{type(exc).__name__}: {exc}")