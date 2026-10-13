"""
LLM client behaviour with no credentials.

The agent's guarantees must not depend on the LLM being available, so
these tests pin the degradation paths rather than the happy path. No
network call is made anywhere in this file.
"""

import unittest
from unittest.mock import MagicMock, patch

from src import llm


def fake_response(text="", stop_reason="end_turn"):
    response = MagicMock()
    block = MagicMock()
    block.type = "text"
    block.text = text
    response.content = [block]
    response.stop_reason = stop_reason
    response.usage = MagicMock(input_tokens=10, output_tokens=20)
    return response


class AvailabilityTests(unittest.TestCase):
    def test_reports_unavailable_without_the_sdk(self):
        with patch("src.llm._sdk", side_effect=llm.LLMUnavailable("no sdk")):
            self.assertFalse(llm.is_available())
            self.assertEqual(llm.unavailable_reason(), "no sdk")

    def test_missing_credentials_are_reported_not_raised(self):
        with patch("src.llm._sdk", return_value=MagicMock()):
            with patch(
                "src.llm.os.getenv", return_value=None
            ):
                with patch(
                    "src.llm._sdk", return_value=MagicMock(
                        Anthropic=MagicMock(
                            side_effect=Exception("no credentials")
                        )
                    )
                ):
                    self.assertFalse(llm.is_available())
                    self.assertIn("credentials", llm.unavailable_reason())

    def test_is_available_true_when_client_constructs(self):
        sdk = MagicMock()
        sdk.Anthropic.return_value = MagicMock()
        with patch("src.llm._sdk", return_value=sdk):
            self.assertTrue(llm.is_available())
            self.assertIsNone(llm.unavailable_reason())


class CompleteJsonTests(unittest.TestCase):
    def _call(self, response, **overrides):
        client = MagicMock()
        client.messages.create.return_value = response
        kwargs = dict(
            schema={"type": "object"},
            system="s",
            user="u",
        )
        kwargs.update(overrides)
        with patch("src.llm._client", return_value=client):
            return llm.complete_json(**kwargs), client

    def test_returns_parsed_json(self):
        result, client = self._call(fake_response('{"facts": []}'))
        self.assertEqual(result.data, {"facts": []})
        self.assertEqual(result.output_tokens, 20)

    def test_uses_the_documented_request_shape(self):
        _, client = self._call(fake_response("{}"))
        kwargs = client.messages.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "claude-opus-5")
        self.assertEqual(kwargs["thinking"], {"type": "adaptive"})
        self.assertEqual(
            kwargs["output_config"]["format"],
            {"type": "json_schema", "schema": {"type": "object"}},
        )
        self.assertEqual(kwargs["output_config"]["effort"], "high")

    def test_refusal_is_raised(self):
        with self.assertRaises(llm.LLMRefused):
            self._call(fake_response("", stop_reason="refusal"))

    def test_truncation_is_raised_not_partially_parsed(self):
        with self.assertRaises(llm.LLMTruncated):
            self._call(
                fake_response('{"facts": [{"value":', stop_reason="max_tokens")
            )

    def test_invalid_json_is_raised(self):
        with self.assertRaises(llm.LLMCallFailed):
            self._call(fake_response("not json at all"))

    def test_empty_content_is_raised(self):
        with self.assertRaises(llm.LLMCallFailed):
            self._call(fake_response(""))

    def test_transport_failure_is_wrapped(self):
        client = MagicMock()
        client.messages.create.side_effect = RuntimeError("boom")
        with patch("src.llm._client", return_value=client):
            with self.assertRaises(llm.LLMCallFailed):
                llm.complete_json(schema={}, system="s", user="u")


class CompleteTextTests(unittest.TestCase):
    def test_returns_text(self):
        client = MagicMock()
        client.messages.create.return_value = fake_response("A briefing.")
        with patch("src.llm._client", return_value=client):
            self.assertEqual(
                llm.complete_text(system="s", user="u"), "A briefing."
            )

    def test_no_schema_constraint_on_text_calls(self):
        client = MagicMock()
        client.messages.create.return_value = fake_response("ok")
        with patch("src.llm._client", return_value=client):
            llm.complete_text(system="s", user="u")
        kwargs = client.messages.create.call_args.kwargs
        self.assertNotIn("format", kwargs.get("output_config", {}))


if __name__ == "__main__":
    unittest.main()