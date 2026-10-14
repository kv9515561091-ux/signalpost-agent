"""
Rate limiting.

The limiter is untestable if it actually sleeps, so the clock and the
sleeper are injected. These tests assert the spacing arithmetic and
that pacing can be turned off, not that time passes.
"""

import os
import unittest
from unittest import mock
from unittest.mock import patch

from src.ratelimit import (
    DEFAULT_RPM,
    MAX_RPM,
    MIN_RPM,
    RateLimiter,
    requests_per_minute,
)


class FakeClock:
    """A clock that only advances when something sleeps."""

    def __init__(self):
        self.now = 1000.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


class ConfigurationTests(unittest.TestCase):
    def test_default_is_below_the_observed_ceiling(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(requests_per_minute(), DEFAULT_RPM)
        self.assertLess(DEFAULT_RPM, 40)

    def test_env_var_overrides(self):
        with patch.dict(os.environ, {"NVIDIA_RPM": "12"}):
            self.assertEqual(requests_per_minute(), 12)

    def test_garbage_falls_back_to_the_default(self):
        with patch.dict(os.environ, {"NVIDIA_RPM": "not-a-number"}):
            self.assertEqual(requests_per_minute(), DEFAULT_RPM)

    def test_values_are_clamped(self):
        with patch.dict(os.environ, {"NVIDIA_RPM": "0"}):
            self.assertEqual(requests_per_minute(), MIN_RPM)
        with patch.dict(os.environ, {"NVIDIA_RPM": "99999"}):
            self.assertEqual(requests_per_minute(), MAX_RPM)


class IntervalTests(unittest.TestCase):
    def test_interval_derives_from_rpm(self):
        limiter = RateLimiter(rpm=60)
        self.assertAlmostEqual(limiter.min_interval, 1.0)

    def test_zero_rpm_disables_pacing(self):
        clock = FakeClock()
        limiter = RateLimiter(rpm=0, clock=clock.monotonic, sleeper=clock.sleep)
        for _ in range(5):
            limiter.acquire()
        self.assertEqual(clock.slept, [])

    def test_first_call_does_not_wait(self):
        clock = FakeClock()
        limiter = RateLimiter(rpm=60, clock=clock.monotonic, sleeper=clock.sleep)
        self.assertEqual(limiter.acquire(), 0.0)
        self.assertEqual(clock.slept, [])


class SpacingTests(unittest.TestCase):
    def _limiter(self, rpm=60):
        clock = FakeClock()
        limiter = RateLimiter(rpm=rpm, clock=clock.monotonic, sleeper=clock.sleep)
        return limiter, clock

    def test_calls_are_spaced_by_the_interval(self):
        limiter, clock = self._limiter(rpm=60)  # 1 second apart
        limiter.acquire()
        limiter.acquire()
        limiter.acquire()
        self.assertEqual([round(s, 3) for s in clock.slept], [1.0, 1.0])

    def test_elapsed_time_reduces_the_wait(self):
        limiter, clock = self._limiter(rpm=60)
        limiter.acquire()
        clock.now += 0.6      # caller spent 600ms elsewhere
        waited = limiter.acquire()
        self.assertAlmostEqual(waited, 0.4, places=3)

    def test_enough_elapsed_time_means_no_wait(self):
        limiter, clock = self._limiter(rpm=60)
        limiter.acquire()
        clock.now += 5.0
        self.assertEqual(limiter.acquire(), 0.0)
        self.assertEqual(clock.slept, [])

    def test_sustained_rate_stays_within_the_ceiling(self):
        # Ten calls at 30 RPM (2s apart) must span at least 18 seconds.
        limiter, clock = self._limiter(rpm=30)
        start = clock.now
        for _ in range(10):
            limiter.acquire()
        self.assertGreaterEqual(clock.now - start, 18.0)

    def test_total_wait_is_reported(self):
        limiter, _ = self._limiter(rpm=60)
        limiter.acquire()
        limiter.acquire()
        self.assertAlmostEqual(limiter.total_waited, 1.0, places=3)

    def test_reset_clears_state(self):
        limiter, clock = self._limiter(rpm=60)
        limiter.acquire()
        limiter.reset()
        limiter.acquire()
        self.assertEqual(len(clock.slept), 0)


class BackendIntegrationTests(unittest.TestCase):
    def test_backend_paces_through_the_shared_limiter(self):
        from src import llm_openai
        from src.ratelimit import RateLimiter

        clock = FakeClock()
        limiter = RateLimiter(rpm=60, clock=clock.monotonic, sleeper=clock.sleep)
        backend = llm_openai.OpenAICompatibleBackend(
            api_key="k", model="m", rate_limiter=limiter
        )

        ok = mock.MagicMock()
        ok.status_code = 200
        ok.json.return_value = {
            "choices": [{"finish_reason": "stop", "message": {"content": "x"}}]
        }
        ok.raise_for_status = lambda: None

        with mock.patch("src.llm_openai.requests.post", mock.MagicMock(return_value=ok)):
            backend.complete_text(system="s", user="u")
            backend.complete_text(system="s", user="u")

        self.assertEqual(len(clock.slept), 1, "second call should be paced")

    def test_default_backend_shares_the_process_limiter(self):
        from src import llm_openai
        from src.ratelimit import limiter as shared

        backend = llm_openai.OpenAICompatibleBackend(api_key="k", model="m")
        self.assertIs(backend.rate_limiter, shared)


if __name__ == "__main__":
    unittest.main()
