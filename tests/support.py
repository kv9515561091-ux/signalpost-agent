"""
Shared test helpers.

The agent caches registry responses on disk so repeated runs do not
re-fetch. That is a problem in tests: a cached body would satisfy an
assertion before the mocked transport is ever called, and a test run
would leave files in the working tree.

Anything that exercises HTTP therefore redirects the cache to a
temporary directory for the duration of the test.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


def disable_llm_pacing():
    """Disable request pacing for the duration of a test module.

    The limiter exists to stay under a real provider's quota. Sleeping
    between mocked calls turns a ten-second suite into a minute without
    testing anything, so every module that exercises the LLM path turns
    it off.
    """
    from src import llm_openai
    from src.ratelimit import RateLimiter

    original = llm_openai.limiter
    llm_openai.limiter = RateLimiter(rpm=0)
    return original


def restore_llm_pacing(original):
    from src import llm_openai

    llm_openai.limiter = original


class IsolatedCacheMixin:
    """Point the module-level cache at a throwaway directory."""

    def setUp(self) -> None:  # noqa: N802 - unittest naming
        super().setUp()  # type: ignore[misc]
        from src import cache as cache_module

        self._cache_tmp = tempfile.TemporaryDirectory()
        self._cache_previous_dir = cache_module.cache.directory
        self._cache_previous_stats = dict(cache_module.cache.stats())

        cache_module.cache.directory = Path(self._cache_tmp.name)

    def tearDown(self) -> None:  # noqa: N802
        from src import cache as cache_module

        cache_module.cache.directory = self._cache_previous_dir
        for key, value in self._cache_previous_stats.items():
            setattr(cache_module.cache, key, value)

        self._cache_tmp.cleanup()
        super().tearDown()  # type: ignore[misc]


class CacheTestCase(IsolatedCacheMixin, unittest.TestCase):
    """Base class for tests that reach the HTTP layer."""