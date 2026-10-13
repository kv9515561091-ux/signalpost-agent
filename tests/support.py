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