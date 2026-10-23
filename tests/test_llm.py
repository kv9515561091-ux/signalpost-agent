"""
LLM facade: provider selection and error translation.

With two backends, the questions that matter are: which one gets used,
and does a failure from either surface as a clear, actionable message
rather than a stack trace?
"""

import unittest
from unittest.mock import MagicMock, patch

from src import llm


def anthropic_response(text="{}", stop_reason="end_turn"):
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    response.stop_reason = stop_reason
    usage = MagicMock(input_tokens=10, output_tokens=20)
    response.usage = usage
    return response


def anthropic_sdk(response):
    client = MagicMock()
    client.Anthropic.return_value.messages.create.return_value = response
    return MagicMock(return_value=client)


class ProviderSelectionTests(unittest.TestCase):
    """Provider choice must follow configured credentials.

    The Anthropic SDK resolves credentials from an on-disk profile even
    with no environment variable set, so a machine with Claude Code
    installed would otherwise use Claude despite the user configuring
    NVIDIA. Every test here clears the ambient environment for that
    reason.
    """

    def _env(self, **values):
        import os as _os
        return patch.dict(_os.environ, values, clear=True)

    def test_anthropic_wins_when_both_configured(self):
        sdk = anthropic_sdk(anthropic_response())
        with patch.object(llm, "_anthropic_sdk", sdk):
            with self._env(ANTHROPIC_API_KEY="sk-ant-x", NVIDIA_API_KEY="nvapi-x"):
                self.assertEqual(llm.provider_name(), "anthropic")

    def test_configured_nvidia_key_beats_a_discoverable_sdk_profile(self):
        # The real bug: SDK present with an ambient credential, only
        # NVIDIA configured. NVIDIA must win.
        sdk = anthropic_sdk(anthropic_response())
        with patch.object(llm, "_anthropic_sdk", sdk):
            with self._env(NVIDIA_API_KEY="nvapi-x"):
                self.assertEqual(
                    llm.provider_name(), "openai_compatible"
                )
                self.assertIn("nvidia", llm.describe())

    def test_nothing_configured_falls_back_to_a_discoverable_profile(self):
        sdk = anthropic_sdk(anthropic_response())
        with patch.object(llm, "_anthropic_sdk", sdk):
            with self._env():
                self.assertEqual(llm.provider_name(), "anthropic")

    def test_discoverable_profile_is_not_explicit_configuration(self):
        # run_daily must stay keyless on a machine where the Anthropic
        # SDK is installed but no key was supplied.
        sdk = anthropic_sdk(anthropic_response())
        with patch.object(llm, "_anthropic_sdk", sdk):
            with self._env():
                self.assertTrue(llm.is_available())
                self.assertFalse(llm.is_explicitly_configured())

    def test_nvidia_key_is_explicit_configuration(self):
        sdk = anthropic_sdk(anthropic_response())
        with patch.object(llm, "_anthropic_sdk", sdk):
            with self._env(NVIDIA_API_KEY="nvapi-x"):
                self.assertTrue(llm.is_explicitly_configured())

    def test_provider_override_is_explicit_configuration(self):
        sdk = anthropic_sdk(anthropic_response())
        with patch.object(llm, "_anthropic_sdk", sdk):
            with self._env(LLM_PROVIDER="anthropic"):
                self.assertTrue(llm.is_explicitly_configured())
            with self._env(LLM_PROVIDER="none"):
                self.assertFalse(llm.is_explicitly_configured())

    def test_no_credentials_and_no_sdk_is_unavailable(self):
        with patch.object(
            llm, "_anthropic_sdk", side_effect=llm.LLMUnavailable("no sdk")
        ):
            with self._env():
                self.assertFalse(llm.is_available())

    def test_override_forces_nvidia(self):
        sdk = anthropic_sdk(anthropic_response())
        with patch.object(llm, "_anthropic_sdk", sdk):
            with self._env(
                ANTHROPIC_API_KEY="sk-ant-x",
                NVIDIA_API_KEY="nvapi-x",
                LLM_PROVIDER="nvidia",
            ):
                self.assertEqual(
                    llm.provider_name(), "openai_compatible"
                )

    def test_override_none_disables(self):
        with self._env(LLM_PROVIDER="none", NVIDIA_API_KEY="nvapi-x"):
            self.assertFalse(llm.is_available())
            self.assertIn("disabled", llm.unavailable_reason())

    def test_no_keys_reports_how_to_configure(self):
        with patch.object(
            llm, "_anthropic_sdk", side_effect=llm.LLMUnavailable("no sdk")
        ):
            with self._env():
                reason = llm.unavailable_reason()
                self.assertIn("ANTHROPIC_API_KEY", reason)
                self.assertIn("NVIDIA_API_KEY", reason)

    def test_anthropic_key_without_sdk_names_the_install(self):
        with patch.object(
            llm, "_anthropic_sdk", side_effect=llm.LLMUnavailable("no sdk")
        ):
            with self._env(ANTHROPIC_API_KEY="sk-ant-x"):
                self.assertIn(
                    "requirements-llm.txt", llm.unavailable_reason()
                )

    def test_describe_is_human_readable(self):
        with self._env(LLM_PROVIDER="none"):
            self.assertEqual(llm.describe(), "none")


class AnthropicDispatchTests(unittest.TestCase):
    def _call_json(self, response):
        sdk = anthropic_sdk(response)
        with patch.object(llm, "_anthropic_sdk", sdk):
            with patch.object(
                llm, "_selected_provider", return_value="anthropic"
            ):
                return llm.complete_json(
                    schema={"type": "object"}, system="s", user="u"
                )

    def test_returns_parsed_json_and_token_counts(self):
        result = self._call_json(anthropic_response('{"facts": []}'))
        self.assertEqual(result.data, {"facts": []})
        self.assertEqual(result.output_tokens, 20)
        self.assertEqual(result.total_tokens, 30)
        self.assertEqual(result.provider, "anthropic")

    def test_request_shape_is_the_documented_one(self):
        sdk = anthropic_sdk(anthropic_response("{}"))
        with patch.object(llm, "_anthropic_sdk", sdk):
            with patch.object(
                llm, "_selected_provider", return_value="anthropic"
            ):
                llm.complete_json(
                    schema={"type": "object"}, system="s", user="u"
                )
        kwargs = sdk.return_value.Anthropic.return_value.messages.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "claude-opus-5")
        self.assertEqual(kwargs["thinking"], {"type": "adaptive"})
        self.assertEqual(
            kwargs["output_config"]["format"],
            {"type": "json_schema", "schema": {"type": "object"}},
        )
        self.assertEqual(kwargs["output_config"]["effort"], "high")
        self.assertGreater(kwargs["max_tokens"], 0)

    def test_text_call_has_no_schema_constraint(self):
        sdk = anthropic_sdk(anthropic_response("A briefing."))
        with patch.object(llm, "_anthropic_sdk", sdk):
            with patch.object(
                llm, "_selected_provider", return_value="anthropic"
            ):
                self.assertEqual(
                    llm.complete_text(system="s", user="u"), "A briefing."
                )
        kwargs = sdk.return_value.Anthropic.return_value.messages.create.call_args.kwargs
        self.assertNotIn("format", kwargs["output_config"])

    def test_refusal_is_raised(self):
        with self.assertRaises(llm.LLMRefused):
            self._call_json(anthropic_response("", stop_reason="refusal"))

    def test_truncation_is_raised_not_parsed(self):
        with self.assertRaises(llm.LLMTruncated):
            self._call_json(
                anthropic_response('{"facts":', stop_reason="max_tokens")
            )

    def test_invalid_json_is_raised(self):
        with self.assertRaises(llm.LLMCallFailed):
            self._call_json(anthropic_response("not json"))

    def test_empty_content_is_raised(self):
        with self.assertRaises(llm.LLMCallFailed):
            self._call_json(anthropic_response(""))

    def test_transport_failure_is_wrapped(self):
        client = MagicMock()
        client.Anthropic.return_value.messages.create.side_effect = (
            RuntimeError("boom")
        )
        with patch.object(
            llm, "_anthropic_sdk", MagicMock(return_value=client)
        ):
            with patch.object(
                llm, "_selected_provider", return_value="anthropic"
            ):
                with self.assertRaises(llm.LLMCallFailed):
                    llm.complete_json(schema={}, system="s", user="u")


class OpenAICompatibleDispatchTests(unittest.TestCase):
    def test_backend_result_is_returned_with_usage(self):
        backend = MagicMock()
        backend.model = "nvidia/nemotron-3-super-120b-a12b"
        backend.complete_json.return_value = {"facts": []}
        backend.last_usage = {
            "prompt_tokens": 100, "completion_tokens": 50
        }

        with patch.object(llm, "_selected_provider", return_value="openai_compatible"):
            with patch.object(llm, "_openai_backend", return_value=backend):
                result = llm.complete_json(
                    schema={}, system="s", user="u"
                )

        self.assertEqual(result.data, {"facts": []})
        self.assertEqual(result.input_tokens, 100)
        self.assertEqual(result.output_tokens, 50)
        self.assertEqual(result.provider, "openai_compatible")

    def test_backend_errors_are_translated(self):
        from src.llm_openai import CallFailed, Refused, Truncated

        backend = MagicMock(model="m", last_usage={})
        backend.complete_json.side_effect = Refused("key bad")
        with patch.object(llm, "_selected_provider", return_value="openai_compatible"):
            with patch.object(llm, "_openai_backend", return_value=backend):
                with self.assertRaises(llm.LLMRefused):
                    llm.complete_json(schema={}, system="s", user="u")

        backend.complete_json.side_effect = Truncated("cut off")
        with patch.object(llm, "_selected_provider", return_value="openai_compatible"):
            with patch.object(llm, "_openai_backend", return_value=backend):
                with self.assertRaises(llm.LLMTruncated):
                    llm.complete_json(schema={}, system="s", user="u")

        backend.complete_json.side_effect = CallFailed("timeout")
        with patch.object(llm, "_selected_provider", return_value="openai_compatible"):
            with patch.object(llm, "_openai_backend", return_value=backend):
                with self.assertRaises(llm.LLMCallFailed):
                    llm.complete_json(schema={}, system="s", user="u")

    def test_unknown_error_is_still_wrapped(self):
        backend = MagicMock(model="m", last_usage={})
        backend.complete_json.side_effect = ValueError("odd")
        with patch.object(llm, "_selected_provider", return_value="openai_compatible"):
            with patch.object(llm, "_openai_backend", return_value=backend):
                with self.assertRaises(llm.LLMCallFailed):
                    llm.complete_json(schema={}, system="s", user="u")


if __name__ == "__main__":
    unittest.main()
