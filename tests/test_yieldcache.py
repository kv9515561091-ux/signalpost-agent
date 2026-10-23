"""Spending the time budget where it earns facts.

The failure this guards against is not an exception: a run that reads
known-empty companies first and runs out of clock before reaching the
productive ones looks perfectly healthy and scores badly. Order and
depth are the behaviour under test.
"""

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src import batch, yieldcache
from src.agent import DEFAULT_MAX_PAGES


class FakeProfile:
    def __init__(self, number="123456789", facts=()):
        self.company_number = number
        self.facts = list(facts)
        self._data = {
            "company_number": number,
            "company_name": "EXAMPLE AS",
            "facts": [],
            "explanation": None,
            "financials": None,
            "ceo_name": None,
        }
        if self.facts:
            self._data["facts"] = [{"field": "website"}]

    def model_dump(self, mode="json"):
        return dict(self._data)


def fact(url):
    return SimpleNamespace(source_url=url)


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "cache.json"
        yieldcache.use_path(self.path)
        self.addCleanup(yieldcache.reset)

    def test_record_persists_across_reload(self):
        yieldcache.record("111111111", facts=4, domains={"a.no", "b.no"})

        yieldcache.reset()
        self.assertEqual(yieldcache.facts_last_time("111111111"), 4)
        self.assertEqual(
            sorted(yieldcache.domains_last_time("111111111")), ["a.no", "b.no"]
        )

        stored = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(stored["111111111"]["facts"], 4)

    def test_priority_bands(self):
        yieldcache.record("111111111", facts=3)
        yieldcache.record("222222222", facts=0)

        self.assertEqual(
            yieldcache.priority("111111111"), yieldcache.PRIORITY_PRODUCTIVE
        )
        self.assertEqual(yieldcache.priority("333333333"), yieldcache.PRIORITY_UNKNOWN)
        self.assertEqual(yieldcache.priority("222222222"), yieldcache.PRIORITY_EMPTY)

    def test_disabled_cache_learns_nothing(self):
        with patch.dict(os.environ, {"AGENT_YIELD_CACHE": "0"}):
            yieldcache.record("111111111", facts=0)
            self.assertEqual(
                yieldcache.priority("111111111"), yieldcache.PRIORITY_UNKNOWN
            )
            self.assertFalse(self.path.exists())


class OrderingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name) / "out.json"
        yieldcache.use_path(Path(self.tmp.name) / "cache.json")
        self.addCleanup(yieldcache.reset)
        self.calls = []

    def _run(self, numbers, **kwargs):
        def fake_build(raw, **kw):
            self.calls.append(raw)
            return FakeProfile(number=raw)

        with patch.object(batch, "build_profile", fake_build):
            return batch.process_companies(
                list(numbers),
                output_file=str(self.out),
                delay=0,
                include_extraction=True,
                **kwargs,
            )

    def test_productive_companies_are_worked_first_and_empty_ones_last(self):
        yieldcache.record("111111111", facts=0)
        yieldcache.record("222222222", facts=6)
        yieldcache.record("333333333", facts=2)

        results = self._run(["111111111", "333333333", "222222222", "444444444"])

        # Productive (333, 222 -- input order inside the band) before
        # unknown (444) before empty (111).
        self.assertEqual(
            self.calls, ["333333333", "222222222", "444444444", "111111111"]
        )
        # The output file still follows the input order.
        self.assertEqual(
            [row["company_number"] for row in results],
            ["111111111", "333333333", "222222222", "444444444"],
        )

    def test_an_empty_company_gets_a_shallower_walk(self):
        yieldcache.record("111111111", facts=0)
        depths = {}

        def fake_build(raw, **kw):
            depths[raw] = kw["max_pages"]
            return FakeProfile(number=raw)

        with patch.object(batch, "build_profile", fake_build):
            batch.process_companies(
                ["111111111", "222222222"],
                output_file=str(self.out),
                delay=0,
                include_extraction=True,
            )

        self.assertEqual(depths["111111111"], yieldcache.ZERO_YIELD_PAGES)
        self.assertEqual(depths["222222222"], DEFAULT_MAX_PAGES)

    def test_outcome_is_cached_for_the_next_run(self):
        def fake_build(raw, **kw):
            return FakeProfile(
                number=raw, facts=[fact("https://a.no/x"), fact("https://a.no/y")]
            )

        with patch.object(batch, "build_profile", fake_build):
            batch.process_companies(
                ["111111111"],
                output_file=str(self.out),
                delay=0,
                include_extraction=True,
            )

        yieldcache.reset()
        self.assertEqual(yieldcache.facts_last_time("111111111"), 2)
        self.assertEqual(yieldcache.domains_last_time("111111111"), ["a.no"])


class DeadlineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name) / "out.json"
        yieldcache.use_path(Path(self.tmp.name) / "cache.json")
        self.addCleanup(yieldcache.reset)

    def test_a_passed_deadline_builds_from_the_registry_alone(self):
        seen = {}

        def fake_build(raw, **kw):
            seen.update(kw)
            return FakeProfile(number=raw)

        with patch.object(batch, "build_profile", fake_build):
            results = batch.process_companies(
                ["111111111"],
                output_file=str(self.out),
                delay=0,
                include_extraction=True,
                deadline=time.monotonic() - 1,
            )

        self.assertFalse(seen["include_web"])
        self.assertFalse(seen["include_extraction"])
        self.assertIn(
            "daily time budget",
            " ".join(results[0]["retrieval_errors"]),
        )

    def test_a_truncated_company_is_retried_by_the_next_run(self):
        calls = []

        def fake_build(raw, **kw):
            calls.append(kw["include_web"])
            return FakeProfile(number=raw)

        with patch.object(batch, "build_profile", fake_build):
            batch.process_companies(
                ["111111111"],
                output_file=str(self.out),
                delay=0,
                include_extraction=True,
                deadline=time.monotonic() - 1,
            )
            # Resume must not treat the shallow record as finished.
            batch.process_companies(
                ["111111111"],
                output_file=str(self.out),
                delay=0,
                include_extraction=True,
                resume=True,
            )

        self.assertEqual(calls, [False, True])

    def test_a_skipped_company_is_not_taught_as_empty(self):
        with patch.object(
            batch, "build_profile", lambda raw, **kw: FakeProfile(number=raw)
        ):
            batch.process_companies(
                ["111111111"],
                output_file=str(self.out),
                delay=0,
                include_extraction=True,
                deadline=time.monotonic() - 1,
            )

        yieldcache.reset()
        self.assertIsNone(yieldcache.facts_last_time("111111111"))
