"""
Search caching.

The hazard these tests exist for: ``cached_json`` keys on URL alone, and
every TinyFish query goes to the same endpoint with the query in the
query string. Wiring the cache up naively would serve one company's
results to every other company — a silent, expensive, wrong-answer bug.

So the tests here assert *isolation* first and cost second.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import cache as cache_module
from src import tinyfish_search
from src.cache import SEARCH_TTL_SECONDS, DiskCache, cached_json


def _payload(url="https://example.no"):
    return {
        "results": [
            {"url": url, "title": "Example", "snippet": "text"},
        ]
    }


class SearchUrlTests(unittest.TestCase):
    def test_query_is_part_of_the_url(self):
        url = tinyfish_search.build_search_url("ACME AS")
        self.assertIn("query=ACME", url)

    def test_different_queries_produce_different_urls(self):
        a = tinyfish_search.build_search_url("ACME AS")
        b = tinyfish_search.build_search_url("BETA AS")
        self.assertNotEqual(a, b)

    def test_localisation_parameters_are_carried(self):
        url = tinyfish_search.build_search_url("ACME AS")
        self.assertIn("location=NO", url)
        self.assertIn("language=no", url)

    def test_the_key_is_never_the_bare_endpoint(self):
        url = tinyfish_search.build_search_url("ACME AS")
        self.assertNotEqual(url, tinyfish_search.TINYFISH_SEARCH_URL)

    def test_credential_is_never_in_the_url(self):
        with patch.dict(os.environ, {"TINYFISH_API_KEY": "secret-key"}):
            url = tinyfish_search.build_search_url("ACME AS")
        self.assertNotIn("secret-key", url)


class SearchCacheIsolationTests(unittest.TestCase):
    """Each query must get its own entry and its own answer."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        patcher = patch.object(
            cache_module,
            "cache",
            DiskCache(self.dir, cache_module.DEFAULT_TTL_SECONDS),
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        cache_module._caches.clear()
        self.addCleanup(cache_module._caches.clear)

    def test_distinct_queries_do_not_collide(self):
        calls = []

        def fake_get(url, **kwargs):
            calls.append(url)
            query = "ACME" if "ACME" in url else "BETA"
            body = _payload(f"https://{query.lower()}.no")
            response = _FakeResponse(body)
            return response

        with patch.dict(os.environ, {"TINYFISH_API_KEY": "k"}), patch.object(
            tinyfish_search.requests, "get", fake_get
        ):
            first = tinyfish_search.search_tinyfish("ACME AS")
            second = tinyfish_search.search_tinyfish("BETA AS")

        self.assertEqual(len(calls), 2, "each query needs its own request")
        self.assertEqual(first[0]["url"], "https://acme.no")
        self.assertEqual(second[0]["url"], "https://beta.no")

    def test_repeat_query_is_served_from_cache(self):
        calls = []

        def fake_get(url, **kwargs):
            calls.append(url)
            return _FakeResponse(_payload())

        with patch.dict(os.environ, {"TINYFISH_API_KEY": "k"}), patch.object(
            tinyfish_search.requests, "get", fake_get
        ):
            tinyfish_search.search_tinyfish("ACME AS")
            tinyfish_search.search_tinyfish("ACME AS")

        self.assertEqual(len(calls), 1, "second identical query should be free")

    def test_max_results_does_not_break_cache_reuse(self):
        calls = []

        def fake_get(url, **kwargs):
            calls.append(url)
            return _FakeResponse(
                {
                    "results": [
                        {"url": f"https://e{i}.no", "title": "t", "snippet": "s"}
                        for i in range(10)
                    ]
                }
            )

        with patch.dict(os.environ, {"TINYFISH_API_KEY": "k"}), patch.object(
            tinyfish_search.requests, "get", fake_get
        ):
            many = tinyfish_search.search_tinyfish("ACME AS", max_results=10)
            few = tinyfish_search.search_tinyfish("ACME AS", max_results=3)

        self.assertEqual(len(calls), 1, "trimming must not force a refetch")
        self.assertEqual(len(many), 10)
        self.assertEqual(len(few), 3)

    def test_credentials_are_sent_but_not_stored(self):
        def fake_get(url, **kwargs):
            self.assertEqual(kwargs["headers"]["X-API-Key"], "k")
            return _FakeResponse(_payload())

        with patch.dict(os.environ, {"TINYFISH_API_KEY": "k"}), patch.object(
            tinyfish_search.requests, "get", fake_get
        ):
            tinyfish_search.search_tinyfish("ACME AS")

        for path in self.dir.glob("*.json"):
            self.assertNotIn("k\"", path.read_text(encoding="utf-8")[:200])
            self.assertNotIn('"k"', path.read_text(encoding="utf-8"))

    def test_bypass_flag_skips_the_cache(self):
        calls = []

        def fake_get(url, **kwargs):
            calls.append(url)
            return _FakeResponse(_payload())

        with patch.dict(os.environ, {"TINYFISH_API_KEY": "k"}), patch.object(
            tinyfish_search.requests, "get", fake_get
        ):
            tinyfish_search.search_tinyfish("ACME AS", use_cache=False)
            tinyfish_search.search_tinyfish("ACME AS", use_cache=False)

        self.assertEqual(len(calls), 2)


class SearchTtlTests(unittest.TestCase):
    def test_search_ttl_is_shorter_than_the_registry(self):
        self.assertLess(SEARCH_TTL_SECONDS, cache_module.DEFAULT_TTL_SECONDS)

    def test_search_ttl_is_an_hour(self):
        self.assertEqual(SEARCH_TTL_SECONDS, 3600)

    def test_expired_search_entry_is_refetched(self):
        with tempfile.TemporaryDirectory() as tmp:
            active = DiskCache(tmp, SEARCH_TTL_SECONDS)
            url = "https://api.search.tinyfish.ai?q=x"
            active.put(url, {"results": []}, "2026-01-01T00:00:00Z")
            self.assertIsNotNone(active.get_fresh(url))

            # Age the entry past its TTL.
            path = active._path(url)
            record = json.loads(path.read_text(encoding="utf-8"))
            record["stored_at"] -= SEARCH_TTL_SECONDS + 60
            path.write_text(json.dumps(record), encoding="utf-8")

            self.assertIsNone(active.get_fresh(url))


class CacheReuseTests(unittest.TestCase):
    def test_cache_for_reuses_one_instance(self):
        cache_module._caches.clear()
        self.addCleanup(cache_module._caches.clear)
        a = cache_module.cache_for("data/cache", 3600)
        b = cache_module.cache_for("data/cache", 3600)
        self.assertIs(a, b)

    def test_different_ttls_get_different_instances(self):
        cache_module._caches.clear()
        self.addCleanup(cache_module._caches.clear)
        a = cache_module.cache_for("data/cache", 3600)
        b = cache_module.cache_for("data/cache", 7200)
        self.assertIsNot(a, b)

    def test_stats_survive_repeated_lookups(self):
        cache_module._caches.clear()
        self.addCleanup(cache_module._caches.clear)
        shared = cache_module.cache_for("data/cache", 900)
        cache_module.cache_for("data/cache", 900).put(
            "https://example.no/a", {"x": 1}, "2026-01-01T00:00:00Z"
        )
        cache_module.cache_for("data/cache", 900).get_fresh("https://example.no/a")
        self.assertEqual(shared.hits, 1)


class _FakeResponse:
    def __init__(self, body):
        self._body = body
        self.status_code = 200
        self.encoding = "utf-8"

    def json(self):
        return self._body

    def raise_for_status(self):
        return None


if __name__ == "__main__":
    unittest.main()