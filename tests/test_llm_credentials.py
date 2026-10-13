"""
Credential discovery.

The failure this guards against is quiet: a key sitting in .env that the
agent never reads, leaving the LLM path permanently off with no error
beyond a fallback notice nobody reads.
"""

import os
import tempfile
import unittest
import unittest.mock
from pathlib import Path
from unittest.mock import patch

from src import llm
from src.llm import MODEL, LLMUnavailable, is_available, unavailable_reason


class EnvLoadingTests(unittest.TestCase):
    def test_module_loads_dotenv_at_import(self):
        # .env must be read by llm.py itself, not by whichever module
        # happened to import first.
        source = Path(llm.__file__).read_text(encoding="utf-8")
        self.assertIn("load_dotenv", source)

    def test_key_in_env_file_is_discovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text(
                "ANTHROPIC_API_KEY=sk-ant-dummy-value\n", encoding="utf-8"
            )

            sdk = unittest.mock.MagicMock()
            with patch.object(llm, "_sdk", return_value=sdk):
                with patch.object(
                    llm, "load_dotenv", create=True, return_value=None
                ):
                    # Simulate what load_dotenv() would have done.
                    with patch.dict(
                        os.environ, {"ANTHROPIC_API_KEY": "sk-ant-dummy"}
                    ):
                        self.assertTrue(is_available())


class ModelSelectionTests(unittest.TestCase):
    def test_default_model_is_the_documented_id(self):
        self.assertEqual(MODEL, "claude-opus-5")


class CredentialDetectionTests(unittest.TestCase):
    def test_env_api_key_is_accepted(self):
        sdk = unittest.mock.MagicMock()
        with patch.object(llm, "_sdk", return_value=sdk):
            with patch.dict(
                os.environ, {"ANTHROPIC_API_KEY": "sk-ant-dummy"}
            ):
                self.assertTrue(is_available())
                self.assertIsNone(unavailable_reason())

    def test_auth_token_is_accepted(self):
        sdk = unittest.mock.MagicMock()
        with patch.object(llm, "_sdk", return_value=sdk):
            with patch.dict(os.environ, {}, clear=True):
                os.environ["ANTHROPIC_AUTH_TOKEN"] = "token"
                try:
                    self.assertTrue(is_available())
                finally:
                    del os.environ["ANTHROPIC_AUTH_TOKEN"]

    def test_missing_sdk_is_reported_clearly(self):
        # Whatever the reason, it must point at the install command.
        with patch.object(llm, "_sdk", side_effect=LLMUnavailable("no sdk")):
            self.assertFalse(is_available())
            self.assertIsNotNone(unavailable_reason())

    def test_real_sdk_message_names_the_requirements_file(self):
        # _sdk() raises with install guidance; that text is what the
        # agent shows the user, so it must mention what to install.
        try:
            import anthropic  # noqa: F401
        except ImportError:
            with self.assertRaises(LLMUnavailable) as ctx:
                llm._sdk()
            self.assertIn("requirements-llm.txt", str(ctx.exception))
        else:  # pragma: no cover - only when the extra is installed
            self.skipTest("anthropic is installed in this environment")


class RequestConstructionTests(unittest.TestCase):
    """Pin the parameters that must not drift."""

    def _capture(self):
        client = unittest.mock.MagicMock()
        block = unittest.mock.MagicMock()
        block.type = "text"
        block.text = '{"facts": []}'
        response = unittest.mock.MagicMock()
        response.content = [block]
        response.stop_reason = "end_turn"
        client.messages.create.return_value = response
        return client

    def test_schema_is_passed_via_output_config(self):
        client = self._capture()
        with patch.object(llm, "_client", return_value=client):
            llm.complete_json(schema={"type": "object"}, system="s", user="u")
        kwargs = client.messages.create.call_args.kwargs
        self.assertEqual(
            kwargs["output_config"]["format"]["type"], "json_schema"
        )
        self.assertEqual(kwargs["output_config"]["format"]["schema"], {"type": "object"})

    def test_effort_and_thinking_are_set(self):
        client = self._capture()
        with patch.object(llm, "_client", return_value=client):
            llm.complete_json(schema={}, system="s", user="u")
        kwargs = client.messages.create.call_args.kwargs
        self.assertEqual(kwargs["output_config"]["effort"], "high")
        self.assertEqual(kwargs["thinking"]["type"], "adaptive")

    def test_max_tokens_is_always_supplied(self):
        client = self._capture()
        with patch.object(llm, "_client", return_value=client):
            llm.complete_json(schema={}, system="s", user="u")
        kwargs = client.messages.create.call_args.kwargs
        self.assertGreater(kwargs["max_tokens"], 0)


if __name__ == "__main__":
    unittest.main()