"""
OpenAI-compatible completion backend.

Used for NVIDIA NIM (``https://integrate.api.nvidia.com/v1``) and any
other endpoint speaking the Chat Completions shape.

Why a bespoke backend rather than the OpenAI SDK: adding that SDK would
pull a dependency whose only job is a handful of POSTs, and the agent
deliberately keeps its runtime to three packages.

The important difference from Claude is that constrained decoding is not
guaranteed here. NVIDIA documents ``guided_json`` for self-hosted NIM
containers but says nothing about the hosted endpoint, and support is
known to vary by model. So this backend:

1. Asks for schema-constrained output when the provider accepts it.
2. Falls back to embedding the schema in the prompt.
3. Extracts JSON defensively, because a model may wrap it in a markdown
   fence, prepend prose, or truncate mid-object.
4. Reports truncation explicitly rather than guessing at a repair.

Callers must still verify whatever comes back. This backend makes the
payload well-formed; it makes nothing true.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import requests

from src.http import RETRY_BACKOFF_SECONDS, time
from src.request_tracker import request_tracker

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Trained explicitly for structured output per NVIDIA's model cards.
DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b"

DEFAULT_TIMEOUT = 120
DEFAULT_MAX_TOKENS = 8_000

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


class Refused(Exception):
    """The provider declined or rejected the request."""


class Truncated(Exception):
    """Generation hit the token limit; output may be incomplete."""


class CallFailed(Exception):
    """Transport, auth or rate-limit failure."""


@dataclass
class Completion:
    text: str
    model: str
    finish_reason: str
    reasoning: str | None = None


def _balanced_json_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` in ``text``.

    Scans with a depth counter while ignoring braces inside strings, so a
    brace appearing in a quoted value does not end the scan early.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start: index + 1]

    return None


def parse_json_response(text: str) -> Any:
    """Parse JSON from a completion, defensively.

    Raises:
        Truncated: The text ends inside an unterminated object.
        CallFailed: No JSON object could be recovered at all.
    """
    candidate = text.strip()

    fenced = _FENCE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    recovered = _balanced_json_object(candidate)
    if recovered:
        try:
            return json.loads(recovered)
        except json.JSONDecodeError:
            pass

    # An unterminated object means the model ran out of room mid-answer.
    if candidate.count("{") > candidate.count("}"):
        raise Truncated("JSON object is unterminated; response was cut off")

    raise CallFailed("no JSON object found in the response")


class OpenAICompatibleBackend:
    """Chat Completions client for OpenAI-compatible endpoints."""

    name = "openai_compatible"

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        thinking: bool = False,
        timeout: int = DEFAULT_TIMEOUT,
        temperature: float = 0.0,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        # Nemotron 3 has reasoning on by default, which consumes the token
        # budget before any answer is written. Extraction needs the answer,
        # not a visible reasoning trace, so it defaults off.
        self.thinking = thinking
        self.timeout = timeout
        # Low temperature matters here: the model must copy a quote
        # verbatim, and at 1.0 it paraphrases, so the quote fails
        # verification and a true fact is discarded. NVIDIA's hosted
        # schemas declare an exclusive minimum of zero for temperature on
        # some models, so a rejection demotes this to 1.0 permanently.
        self.temperature = temperature
        self.fallback_temperature = 1.0
        self._temperature_rejected = False
        # Token counts from the most recent call, so callers can report
        # what a paid request cost.
        self.last_usage: dict[str, Any] = {}

    # ---- transport ---------------------------------------------------
    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None

        for attempt in range(3):
            try:
                request_tracker.record()
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )

                if response.status_code in (401, 403):
                    raise Refused(
                        f"{response.status_code} from provider. Check the "
                        "API key and that it has Public API Endpoints "
                        f"permission. {response.text[:200]}"
                    )
                if response.status_code == 404:
                    raise Refused(
                        f"model {self.model!r} not available for this key "
                        f"({response.text[:200]})"
                    )
                if response.status_code == 429:
                    # NVIDIA does not send Retry-After, and publishes no
                    # per-model limits, so back off conservatively.
                    last_error = CallFailed("rate limited (429)")
                    if attempt < 2:
                        time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 2))
                        continue
                    raise last_error

                if 400 <= response.status_code < 500:
                    # A client error is deterministic. Retrying an
                    # unsupported parameter just spends the budget
                    # three times to reach the same rejection.
                    raise CallFailed(
                        f"{response.status_code} from provider: "
                        f"{response.text[:300]}"
                    )

                response.raise_for_status()
                return response.json()

            except Refused:
                raise
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))

        raise CallFailed(str(last_error) if last_error else "request failed")

    def _base_payload(
        self,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.95,
            "stream": False,
        }

        if not self.thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        return payload

    def _is_parameter_rejection(self, error: CallFailed) -> bool:
        """Whether a failure is about the temperature value specifically.

        Kept narrow on purpose. A 400 can equally mean the endpoint
        rejected ``response_format``, and treating that as a temperature
        problem would burn the schema fallback on the wrong axis.
        """
        text = str(error).lower()

        if any(
            marker in text
            for marker in ("response_format", "json_schema", "guided_json")
        ):
            return False

        return "temperature" in text or any(
            marker in text
            for marker in ("greater than", "exclusive", "less than")
        )

    def _completion_once(
        self,
        system: str,
        user: str,
        max_tokens: int,
        extra: dict[str, Any] | None,
        temperature: float,
    ) -> Completion:
        payload = self._base_payload(system, user, max_tokens, temperature)
        if extra:
            payload.update(extra)

        data = self._post(payload)
        self.last_usage = data.get("usage") or {}

        try:
            choice = data["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise CallFailed(f"unexpected response shape: {str(data)[:200]}") from exc

        text = (message.get("content") or "").strip()
        finish = choice.get("finish_reason") or "stop"

        if finish == "length":
            raise Truncated(
                "generation hit max_tokens; output may be incomplete"
            )
        if not text:
            raise CallFailed("model returned no content")

        return Completion(
            text=text,
            model=data.get("model") or self.model,
            finish_reason=finish,
            reasoning=message.get("reasoning_content"),
        )

    def _completion(
        self,
        system: str,
        user: str,
        max_tokens: int,
        extra: dict[str, Any] | None = None,
    ) -> Completion:
        try:
            return self._completion_once(
                system, user, max_tokens, extra, self.temperature
            )
        except CallFailed as exc:
            if self._temperature_rejected or not self._is_parameter_rejection(exc):
                raise
            # The model will not accept this temperature. Remember that
            # for the rest of the session so it is probed only once.
            self._temperature_rejected = True
            self.temperature = self.fallback_temperature
            return self._completion_once(
                system, user, max_tokens, extra, self.fallback_temperature
            )

    # ---- public surface ----------------------------------------------
    def complete_json(
        self,
        schema: dict[str, Any],
        *,
        system: str,
        user: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Any:
        """Return JSON matching ``schema``, however the provider behaves."""
        constrained_system = (
            system
            + "\n\nRespond with a single JSON object and nothing else. "
            "Do not wrap it in markdown fences or add commentary."
        )

        # Attempt schema-constrained decoding first. Any failure here
        # justifies falling back: the parameter may be unsupported, the
        # model may reject it, or the request may simply have failed.
        # A hard truncation is the exception - retrying would only
        # truncate again - so it propagates.
        try:
            completion = self._completion(
                constrained_system,
                user,
                max_tokens,
                extra={
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "extraction",
                            "schema": schema,
                        },
                    }
                },
            )
            return parse_json_response(completion.text)
        except Truncated:
            raise
        except Exception:  # noqa: BLE001 - fall back by design
            pass

        schema_text = json.dumps(schema, indent=2)
        completion = self._completion(
            constrained_system
            + "\n\nYour entire reply must be JSON conforming to this "
            "schema:\n"
            + schema_text,
            user,
            max_tokens,
        )
        return parse_json_response(completion.text)

    def complete_text(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 2_000,
    ) -> str:
        completion = self._completion(system, user, max_tokens)
        return completion.text

    def describe(self) -> str:
        thinking = "on" if self.thinking else "off"
        return f"{self.name}:{self.model} (thinking {thinking})"


def backend_from_env() -> OpenAICompatibleBackend | None:
    """Build a backend from environment variables, or ``None``."""
    api_key = (
        os.getenv("NVIDIA_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENAI_COMPATIBLE_API_KEY")
    )
    if not api_key:
        return None

    thinking = (os.getenv("NVIDIA_THINKING") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    return OpenAICompatibleBackend(
        api_key=api_key,
        base_url=os.getenv("NVIDIA_BASE_URL") or DEFAULT_BASE_URL,
        model=os.getenv("NVIDIA_MODEL") or DEFAULT_MODEL,
        thinking=thinking,
    )