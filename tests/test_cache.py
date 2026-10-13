"""
Disk cache behaviour.

The cache trades freshness for cost, so the tests that matter are the
ones covering what happens when the cache is wrong: a stale entry must
never be presented as fresh, and a network failure must not silently
return yesterday's data as though it were today's.
"""

import unittest
import unittest.mock

from support import CacheTestCase

from src import cache as cache_module
from src.cache import DiskCache, cached_json, cache
from src.http import NotFound


class DiskCacheTests(CacheTestCase):
    def test_miss_then_hit(self):
        cache.put("https://x/1", {"a": 1}, "2026-10-13T00:00:00+00:00")
        self.assertEqual(cache.get_fresh("https://x/1"), {"a": 1})
        self.assertIsNone(cache.get_fresh("https://x/never-stored"))

    def test_keys_are_scoped_per_url(self):
        cache.put("https://x/a", {"org": "A"}, "t")
        cache.put("https://x/b", {"org": "B"}, "t")
        self.assertEqual(cache.get_fresh("https://x/a"), {"org": "A"})
        self.assertEqual(cache.get_fresh("https://x/b"), {"org": "B"})

    def test_expired_entry_is_not_served_as_fresh(self):
        cache.put("https://x/1", {"old": True}, "t")
        self.assertEqual(cache.get_fresh("https://x/1"), {"old": True})

        short = DiskCache(cache.directory, ttl_seconds=-1)
        self.assertIsNone(short.get_fresh("https://x/1"))
        # ...but it is still recoverable as stale.
        self.assertEqual(short.get_stale("https://x/1"), {"old": True})

    def test_corrupt_entry_is_treated_as_a_miss(self):
        cache.put("https://x/1", {"a": 1}, "t")
        cache._path("https://x/1").write_text("{not json", encoding="utf-8")
        self.assertIsNone(cache.get_fresh("https://x/1"))

    def test_clear_removes_entries(self):
        cache.put("https://x/1", {"a": 1}, "t")
        cache.put("https://x/2", {"b": 2}, "t")
        self.assertEqual(cache.clear(), 2)
        self.assertIsNone(cache.get_fresh("https://x/1"))


class CachedJsonTests(CacheTestCase):
    def test_second_call_is_served_from_cache(self):
        with unittest.mock.patch("src.cache.fetch_json", return_value={"a": 1}) as m:
            first, meta1 = cached_json("https://x/1")
            second, meta2 = cached_json("https://x/1")
        self.assertEqual(first, {"a": 1})
        self.assertEqual(second, {"a": 1})
        self.assertEqual(m.call_count, 1, "second call must not hit network")
        self.assertFalse(meta1["cached"])
        self.assertTrue(meta2["cached"])

    def test_missing_records_are_not_cached(self):
        # A 404 means "no such record", not a body worth storing.
        with unittest.mock.patch("src.cache.fetch_json", return_value=None):
            cached_json("https://x/none", accept_missing=True)
        self.assertEqual(cache.clear(), 0)

    def test_retrieved_at_reflects_the_original_fetch(self):
        with unittest.mock.patch("src.cache.fetch_json", return_value={"a": 1}):
            _, meta = cached_json("https://x/1")
        first_time = meta["retrieved_at"]
        with unittest.mock.patch("src.cache.fetch_json") as m:
            _, meta2 = cached_json("https://x/1")
        m.assert_not_called()
        # Provenance must point at when the data was actually fetched,
        # not when this process happened to read the cache.
        self.assertEqual(meta2["retrieved_at"], first_time)

    def test_stale_entry_serves_when_network_fails(self):
        with unittest.mock.patch("src.cache.fetch_json", return_value={"a": 1}):
            cached_json("https://x/1")

        # A transport failure, not a 404. A 404 is an authoritative
        # "no such record" and must not be papered over with stale data.
        with unittest.mock.patch(
            "src.cache.fetch_json", side_effect=Exception("connection reset")
        ):
            body, meta = cached_json("https://x/1", ttl_seconds=-1)

        self.assertEqual(body, {"a": 1})
        self.assertTrue(meta["stale"], "a stale serve must be flagged")
        self.assertTrue(meta["cached"])

    def test_missing_record_does_not_serve_stale(self):
        # Brreg answered 404; that is the current truth, so a stale body
        # must not be substituted for it.
        with unittest.mock.patch(
            "src.cache.fetch_json", return_value={"a": 1}
        ):
            cached_json("https://x/1")

        with unittest.mock.patch("src.cache.fetch_json", return_value=None):
            body, meta = cached_json("https://x/1", ttl_seconds=-1)

        self.assertIsNone(body)
        self.assertFalse(meta["stale"])

    def test_network_failure_with_no_cache_entry_still_raises(self):
        with unittest.mock.patch(
            "src.cache.fetch_json", side_effect=NotFound("404")
        ):
            with self.assertRaises(NotFound):
                cached_json("https://x/never")

    def test_transport_failure_is_not_cached_as_empty(self):
        with unittest.mock.patch(
            "src.cache.fetch_json", side_effect=Exception("timeout")
        ):
            with self.assertRaises(Exception):
                cached_json("https://x/1")
        self.assertIsNone(cache.get_fresh("https://x/1"))


class AvailabilityTests(unittest.TestCase):
    def test_cache_degrades_when_the_directory_is_unwritable(self):
        # A cache that cannot be written must not break the run: point it
        # at a path whose parent is a regular file, so mkdir cannot work.
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            blocker = Path(tmp) / "not-a-directory"
            blocker.write_text("i am a file", encoding="utf-8")

            broken = DiskCache(blocker / "cache", ttl_seconds=60)
            broken.put("https://x/1", {"a": 1}, "t")  # must not raise
            self.assertIsNone(broken.get_fresh("https://x/1"))


if __name__ == "__main__":
    unittest.main()