"""
OpenAI-compatible backend: request shape, defensive parsing, error paths.

The defensive parser is the point of this backend. Constrained decoding is
not guaranteed on the hosted NVIDIA endpoint and varies by model, so a
model that returns fenced JSON, JSON with prose around it, or truncated
JSON must be handled without ever guessing at the content.
"""

import unittest
from unittest.mock import MagicMock, patch

from support import disable_llm_pacing, restore_llm_pacing

from src.llm_openai import (
    CallFailed,
    OpenAICompatibleBackend,
    Refused,
    Truncated,
    parse_json_response,
)
from src.ratelimit import RateLimiter


def setUpModule():
    global _PACING
    _PACING = disable_llm_pacing()


def tearDownModule():
    restore_llm_pacing(_PACING)


class BalancedObjectTests(unittest.TestCase):
    def test_finds_plain_object(self):
        self.assertEqual(parse_json_response('{"a": 1}'), {"a": 1})

    def test_strips_markdown_fence(self):
        self.assertEqual(
            parse_json_response('```json\n{"a": 1}\n```'), {"a": 1}
        )
        self.assertEqual(
            parse_json_response('```\n{"a": 1}\n```'), {"a": 1}
        )

    def test_recovers_from_surrounding_prose(self):
        text = 'Here is the result:\n{"facts": []}\nHope that helps.'
        self.assertEqual(parse_json_response(text), {"facts": []})

    def test_brace_inside_string_does_not_end_the_object(self):
        text = '{"note": "a } brace", "b": 2}'
        self.assertEqual(parse_json_response(text), {"note": "a } brace", "b": 2})

    def test_escaped_quote_inside_string(self):
        text = '{"note": "say \\"hi\\" }", "b": 2}'
        self.assertEqual(parse_json_response(text)["b"], 2)

    def test_nested_objects_are_matched(self):
        text = '{"facts": [{"field": "x", "quote": "y"}], "n": 1}'
        parsed = parse_json_response(text)
        self.assertEqual(parsed["facts"][0]["field"], "x")

    def test_truncated_object_raises_truncated_not_a_parse_error(self):
        with self.assertRaises(Truncated):
            parse_json_response('{"facts": [{"field": "x"')

    def test_no_json_at_all_raises_call_failed(self):
        with self.assertRaises(CallFailed):
            parse_json_response("I could not find any information.")

    def test_reasoning_prose_is_not_mistaken_for_json(self):
        # Reasoning output arrives separately, but a model may still
        # narrate before the object.
        text = 'Let me consider the page. {"facts": []} Done.'
        self.assertEqual(parse_json_response(text), {"facts": []})


class RequestShapeTests(unittest.TestCase):
    def _backend(self):
        return OpenAICompatibleBackend(
            api_key="nvapi-test",
            model="nvidia/nemotron-3-super-120b-a12b",
        )

    def _response(self, content='{"facts": []}', finish="stop"):
        return {
            "model": "nvidia/nemotron-3-super-120b-a12b",
            "choices": [
                {
                    "finish_reason": finish,
                    "message": {"content": content, "reasoning_content": "…"},
                }
            ],
        }

    def test_uses_bearer_auth_and_json_content_type(self):
        backend = self._backend()
        with patch("src.llm_openai.requests.post") as post:
            post.return_value.status_code = 200
            post.return_value.json.return_value = self._response()
            post.return_value.raise_for_status = lambda: None
            backend.complete_text(system="s", user="u")

        headers = post.call_args.kwargs["headers"]
        self.assertEqual(headers["Authorization"], "Bearer nvapi-test")
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertTrue(
            post.call_args.kwargs["url" if False else "json"] is not None
        )

    def test_posts_to_chat_completions(self):
        backend = self._backend()
        with patch("src.llm_openai.requests.post") as post:
            post.return_value.status_code = 200
            post.return_value.json.return_value = self._response()
            post.return_value.raise_for_status = lambda: None
            backend.complete_text(system="s", user="u")

        self.assertEqual(
            post.call_args[0][0],
            "https://integrate.api.nvidia.com/v1/chat/completions",
        )

    def test_prefers_a_deterministic_temperature(self):
        # A verbatim quote must be copied, not paraphrased; at 1.0 the
        # model rewrites it and a true fact is then discarded.
        backend = self._backend()
        with patch("src.llm_openai.requests.post") as post:
            post.return_value.status_code = 200
            post.return_value.json.return_value = self._response()
            post.return_value.raise_for_status = lambda: None
            backend.complete_text(system="s", user="u")

        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["top_p"], 0.95)

    def test_falls_back_to_one_if_temperature_is_rejected(self):
        backend = self._backend()

        def ok_response(content="ok"):
            response = MagicMock()
            response.status_code = 200
            response.json.return_value = {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": content},
                    }
                ],
                "usage": {},
            }
            response.raise_for_status = lambda: None
            return response

        rejected = MagicMock()
        rejected.status_code = 400
        rejected.text = "temperature must be greater than 0"
        rejected.raise_for_status = MagicMock(
            side_effect=__import__("requests").HTTPError("400")
        )

        post = MagicMock(side_effect=[rejected, ok_response()])
        with patch("src.llm_openai.requests.post", post):
            self.assertEqual(backend.complete_text(system="s", user="u"), "ok")

        self.assertEqual(
            post.call_args_list[-1].kwargs["json"]["temperature"], 1.0
        )

        # The rejection is remembered, so it is never probed twice.
        with patch("src.llm_openai.requests.post") as post2:
            post2.return_value = ok_response()
            backend.complete_text(system="s", user="u")
        self.assertEqual(post2.call_args.kwargs["json"]["temperature"], 1.0)
        self.assertEqual(post2.call_count, 1)

    def test_disables_thinking_by_default(self):
        backend = self._backend()
        with patch("src.llm_openai.requests.post") as post:
            post.return_value.status_code = 200
            post.return_value.json.return_value = self._response()
            post.return_value.raise_for_status = lambda: None
            backend.complete_text(system="s", user="u")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(
            payload["chat_template_kwargs"], {"enable_thinking": False}
        )

    def test_thinking_can_be_enabled(self):
        backend = OpenAICompatibleBackend(
            api_key="k",
            model="m",
            thinking=True,
            rate_limiter=RateLimiter(rpm=0),
        )
        with patch("src.llm_openai.requests.post") as post:
            post.return_value.status_code = 200
            post.return_value.json.return_value = self._response()
            post.return_value.raise_for_status = lambda: None
            backend.complete_text(system="s", user="u")
        payload = post.call_args.kwargs["json"]
        self.assertNotIn("chat_template_kwargs", payload)

    def test_sends_system_and_user_messages(self):
        backend = self._backend()
        with patch("src.llm_openai.requests.post") as post:
            post.return_value.status_code = 200
            post.return_value.json.return_value = self._response()
            post.return_value.raise_for_status = lambda: None
            backend.complete_text(system="be terse", user="hello")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][0]["content"], "be terse")
        self.assertEqual(payload["messages"][1]["role"], "user")


class JsonFallbackTests(unittest.TestCase):
    def _backend(self):
        return OpenAICompatibleBackend(
            api_key="k", model="m", rate_limiter=RateLimiter(rpm=0)
        )

    def _respond(self, responses):
        post = MagicMock()
        post.side_effect = responses
        return post

    def _ok(self, content):
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "choices": [
                {"finish_reason": "stop", "message": {"content": content}}
            ]
        }
        response.raise_for_status = lambda: None
        return response

    def test_falls_back_when_schema_mode_is_rejected(self):
        backend = self._backend()
        bad = MagicMock()
        bad.status_code = 400
        bad.text = "response_format type is unavailable for this model"
        bad.raise_for_status = MagicMock(
            side_effect=__import__("requests").HTTPError("400 Client Error")
        )

        post = self._respond([bad, self._ok('{"facts": []}')])
        with patch("src.llm_openai.requests.post", post):
            with patch("src.llm_openai.time.sleep"):
                data = backend.complete_json(
                    {"type": "object"}, system="s", user="u"
                )
        self.assertEqual(data, {"facts": []})
        # The fallback must embed the schema in the prompt.
        fallback_system = post.call_args_list[-1].kwargs["json"]["messages"][0][
            "content"
        ]
        self.assertIn("schema", fallback_system.lower())
        self.assertNotIn("response_format", post.call_args_list[-1].kwargs["json"])

    def test_truncation_in_constrained_mode_is_not_retried(self):
        # Retrying a truncated constrained request would truncate again.
        backend = self._backend()
        cut = MagicMock()
        cut.status_code = 200
        cut.json.return_value = {
            "choices": [
                {"finish_reason": "length", "message": {"content": "{"}}
            ]
        }
        cut.raise_for_status = lambda: None
        post = MagicMock(return_value=cut)
        with patch("src.llm_openai.requests.post", post):
            with self.assertRaises(Truncated):
                backend.complete_json(
                    {"type": "object"}, system="s", user="u"
                )
        self.assertEqual(post.call_count, 1)

    def test_parses_fenced_json_from_the_primary_path(self):
        backend = self._backend()
        post = MagicMock()
        post.return_value = self._ok('```json\n{"facts": [{"field": "x"}]}\n```')
        with patch("src.llm_openai.requests.post", post):
            data = backend.complete_json(
                {"type": "object"}, system="s", user="u"
            )
        self.assertEqual(data["facts"][0]["field"], "x")
        self.assertEqual(post.call_count, 1, "should not need the fallback")

    def test_truncation_is_reported(self):
        backend = self._backend()
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "choices": [
                {"finish_reason": "length", "message": {"content": "{"}}
            ]
        }
        response.raise_for_status = lambda: None
        with patch("src.llm_openai.requests.post", MagicMock(return_value=response)):
            with self.assertRaises(Truncated):
                backend.complete_text(system="s", user="u")

    def test_auth_failure_is_actionable(self):
        backend = self._backend()
        response = MagicMock()
        response.status_code = 401
        response.text = "Unauthorized"
        post = MagicMock(return_value=response)
        with patch("src.llm_openai.requests.post", post):
            with self.assertRaises(Refused) as ctx:
                backend.complete_text(system="s", user="u")
        self.assertIn("key", str(ctx.exception).lower())

    def test_model_not_available_is_explicit(self):
        backend = self._backend()
        response = MagicMock()
        response.status_code = 404
        response.text = "no such model"
        post = MagicMock(return_value=response)
        with patch("src.llm_openai.requests.post", post):
            with self.assertRaises(Refused) as ctx:
                backend.complete_text(system="s", user="u")
        self.assertIn("not available", str(ctx.exception))

    def test_rate_limit_backs_off_then_fails_cleanly(self):
        backend = self._backend()
        limited = MagicMock(status_code=429, text="slow down")
        post = MagicMock(return_value=limited)
        with patch("src.llm_openai.requests.post", post):
            with patch("src.llm_openai.time.sleep") as sleep:
                with self.assertRaises(CallFailed):
                    backend.complete_text(system="s", user="u")
        self.assertEqual(post.call_count, 3, "should retry before giving up")
        self.assertEqual(
            [call.args[0] for call in sleep.call_args_list],
            [5.0, 10.0],
            "a rate-limit window is longer than the usual retry pause",
        )

    def test_empty_content_is_a_failure(self):
        backend = self._backend()
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "choices": [{"finish_reason": "stop", "message": {"content": ""}}]
        }
        response.raise_for_status = lambda: None
        with patch("src.llm_openai.requests.post", MagicMock(return_value=response)):
            with self.assertRaises(CallFailed):
                backend.complete_text(system="s", user="u")

    def test_malformed_response_shape_is_a_failure(self):
        backend = self._backend()
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"unexpected": True}
        response.raise_for_status = lambda: None
        with patch("src.llm_openai.requests.post", MagicMock(return_value=response)):
            with self.assertRaises(CallFailed):
                backend.complete_text(system="s", user="u")


if __name__ == "__main__":
    unittest.main()
