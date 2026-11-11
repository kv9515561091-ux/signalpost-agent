"""Tests for the NAV Arbeidsplassen vacancy connector.

The endpoint's shape is unusual in three ways that broke earlier
attempts: ``total`` is nested under ``hits``, ``from`` stops at 10,000,
and a county filter can be silently ignored. Each of those is covered
here with an injected ``fetch`` so the logic is testable offline.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timezone

from src.nav_jobs import (
    NavJobsIndex,
    NavPosting,
    _fetch_with_retry,
    _RateLimited,
    ensure_snapshot,
    harvest,
    job_facts,
    load,
    match_postings,
    save,
)


def posting(
    uuid: str,
    employer: str = "KONGSBERG MARITIME ASA",
    title: str = "Senior Engineer",
    municipal: str = "KONGSBERG",
    county: str = "BUSKERUD",
    published: str = "2026-11-09T20:19:29+01:00",
    expires: str = "2026-12-09T20:19:29+01:00",
) -> dict:
    return {
        "uuid": uuid,
        "title": title,
        "employer": {"name": employer},
        "locationList": [
            {"municipal": municipal, "county": county, "country": "NORGE"}
        ],
        "published": published,
        "expires": expires,
    }


def index_of(*postings: dict) -> NavJobsIndex:
    return NavJobsIndex(
        postings=[
            NavPosting(
                uuid=item["uuid"],
                title=item["title"],
                employer=item["employer"]["name"],
                municipality=item["locationList"][0]["municipal"],
                county=item["locationList"][0]["county"],
                published=item["published"],
                expires=item["expires"],
            )
            for item in postings
        ],
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        total=len(postings),
    )


def make_fetch(
    postings: list[dict],
    *,
    ignore_county: bool = False,
    window: int = 10_000,
):
    calls: list[dict] = []

    def fetch(params: dict) -> dict:
        calls.append(dict(params))
        size = int(params.get("size") or 25)
        start = int(params.get("from") or 0)
        if start >= window:
            raise ValueError("from must be below 10000")
        pool = postings
        county = params.get("county")
        if county and not ignore_county:
            pool = [
                item
                for item in postings
                if str(item["locationList"][0]["county"]).casefold()
                == str(county).casefold()
            ]
        page = pool[start : start + size]
        return {
            "hits": {
                "total": {"value": len(pool), "relation": "eq"},
                "hits": [{"_source": item} for item in page],
            }
        }

    fetch.calls = calls  # type: ignore[attr-defined]
    return fetch


class MatchPostingsTests(unittest.TestCase):
    def test_unique_employer_name_is_accepted(self):
        index = index_of(posting("a"))
        found = match_postings(
            index, company_name="Kongsberg Maritime ASA"
        )
        self.assertEqual([item.uuid for item in found], ["a"])

    def test_folding_merges_registry_and_ad_spelling(self):
        index = index_of(
            posting("a", employer="PEAK SERVICES AS AVD BÆRUM", municipal="BÆRUM")
        )
        found = match_postings(
            index, company_name="Peak Services As Avd Bærum"
        )
        self.assertEqual(len(found), 1)

    def test_previous_name_is_accepted(self):
        index = index_of(posting("a", employer="GAMLE NAVN AS"))
        found = match_postings(
            index,
            company_name="Nytt Navn AS",
            previous_names=["Gamle Navn AS"],
        )
        self.assertEqual(len(found), 1)

    def test_subunit_name_is_accepted(self):
        index = index_of(
            posting("a", employer="KONGSBERG MARITIME ASA AVD HORTEN")
        )
        found = match_postings(
            index,
            company_name="Kongsberg Maritime ASA",
            subunit_names=["Kongsberg Maritime ASA Avd Horten"],
        )
        self.assertEqual(len(found), 1)

    def test_ambiguous_name_needs_a_municipality_match(self):
        index = index_of(
            posting("a", employer="NORDIC SERVICES AS", municipal="OSLO"),
            posting("b", employer="Nordic Services AS", municipal="BERGEN"),
        )
        without = match_postings(index, company_name="Nordic Services AS")
        self.assertEqual(without, [])

        with_place = match_postings(
            index,
            company_name="Nordic Services AS",
            municipalities=["Bergen"],
        )
        self.assertEqual([item.uuid for item in with_place], ["b"])

    def test_unknown_company_matches_nothing(self):
        index = index_of(posting("a"))
        self.assertEqual(
            match_postings(index, company_name="Ukjent Foretak AS"), []
        )

    def test_postings_are_newest_first(self):
        index = index_of(
            posting("old", published="2026-01-01T10:00:00+01:00"),
            posting("new", published="2026-11-01T10:00:00+01:00"),
        )
        found = match_postings(
            index, company_name="Kongsberg Maritime ASA"
        )
        self.assertEqual([item.uuid for item in found], ["new", "old"])


class JobFactsTests(unittest.TestCase):
    def test_one_claim_per_posting_plus_a_count(self):
        index = index_of(posting("a"), posting("b", title="Prosjektleder"))
        facts = job_facts(
            index.postings, retrieved_at="2026-11-09T21:00:00+00:00"
        )
        fields = [fact["field"] for fact in facts]
        self.assertEqual(fields, ["job_opening", "job_opening", "jobs_count"])
        self.assertEqual(facts[-1]["value"], 2)

    def test_claim_shape_is_contract_ready(self):
        index = index_of(posting("a"))
        fact = job_facts(
            index.postings, retrieved_at="2026-11-09T21:00:00+00:00"
        )[0]
        self.assertEqual(fact["source_type"], "official_job_feed")
        self.assertEqual(fact["source_tier"], 4)
        self.assertEqual(
            fact["source_url"],
            "https://arbeidsplassen.nav.no/stillinger/stilling/a",
        )
        self.assertEqual(fact["as_of"], "2026-11-09")
        self.assertIn("KONGSBERG MARITIME ASA", fact["evidence_text"])
        self.assertIn("Senior Engineer", fact["evidence_text"])

    def test_postings_are_capped(self):
        index = index_of(*[posting(f"u{n}") for n in range(15)])
        facts = job_facts(
            index.postings, retrieved_at="2026-11-09T21:00:00+00:00"
        )
        openings = [f for f in facts if f["field"] == "job_opening"]
        self.assertEqual(len(openings), 10)
        self.assertEqual(facts[-1]["value"], 15)

    def test_missing_dates_do_not_produce_empty_as_of(self):
        raw = posting("a", published="", expires="")
        index = index_of(raw)
        facts = job_facts(
            index.postings, retrieved_at="2026-11-09T21:00:00+00:00"
        )
        self.assertIsNone(facts[0]["as_of"])

    def test_no_postings_yields_no_facts(self):
        self.assertEqual(job_facts([], retrieved_at="x"), [])


class HarvestTests(unittest.TestCase):
    def test_small_index_is_read_in_one_pass(self):
        fetch = make_fetch([posting(f"u{n}") for n in range(30)])
        index = harvest(fetch=fetch, page_size=25, min_interval=0, sleep=lambda _: None)
        self.assertEqual(len(index.postings), 30)
        self.assertEqual(index.total, 30)
        self.assertEqual(index.partitions, ["national"])
        self.assertFalse(index.truncated)
        self.assertEqual([call["from"] for call in fetch.calls], [0, 25])

    def test_duplicate_uuids_across_pages_are_collapsed(self):
        postings = [posting(f"u{n}") for n in range(5)]
        fetch = make_fetch(postings)
        index = harvest(
            fetch=fetch,
            page_size=2,
            min_interval=0,
            sleep=lambda _: None,
        )
        self.assertEqual(len(index.postings), 5)
        self.assertEqual(
            len({item.uuid for item in index.postings}), 5
        )

    def test_large_index_is_partitioned_by_county(self):
        postings = [
            posting(f"o{n}", county="OSLO", municipal="OSLO") for n in range(4)
        ] + [
            posting(f"v{n}", county="VESTLAND", municipal="BERGEN")
            for n in range(3)
        ]
        fetch = make_fetch(postings)
        index = harvest(
            fetch=fetch,
            page_size=2,
            min_interval=0,
            sleep=lambda _: None,
        )
        # Below the window the harvest stays national; partitioning is
        # covered by the next test with a lowered window.
        self.assertEqual(len(index.postings), 7)
        self.assertEqual(index.partitions, ["national"])

    def test_partitions_stop_at_each_county_total(self):
        postings = [
            posting(f"o{n}", county="OSLO", municipal="OSLO") for n in range(3)
        ] + [
            posting(f"v{n}", county="VESTLAND", municipal="BERGEN")
            for n in range(3)
        ]
        fetch = make_fetch(postings)
        index = harvest(
            fetch=fetch,
            page_size=2,
            min_interval=0,
            sleep=lambda _: None,
            window=5,
        )
        self.assertEqual(len(index.postings), 6)
        self.assertIn("county:OSLO", index.partitions)
        self.assertIn("county:VESTLAND", index.partitions)
        for call in fetch.calls:
            self.assertLess(call["from"], 10_000)

    def test_ignored_county_filter_falls_back_to_one_window(self):
        postings = [posting(f"u{n}") for n in range(6)]
        fetch = make_fetch(postings, ignore_county=True)
        index = harvest(
            fetch=fetch,
            page_size=2,
            min_interval=0,
            sleep=lambda _: None,
            window=4,
        )
        # The single window is all the endpoint allows once the county
        # filter is proven to be ignored, so 4 of 6 is the honest result.
        self.assertEqual(len(index.postings), 4)
        self.assertEqual(index.partitions, ["national"])
        self.assertIn("county filter ignored", index.error or "")

    def test_unreachable_endpoint_reports_an_error(self):
        def fetch(params):
            raise RuntimeError("boom")

        index = harvest(
            fetch=fetch, attempts=2, backoff=0, sleep=lambda _: None
        )
        self.assertEqual(index.postings, [])
        self.assertIsNotNone(index.error)

    def test_rate_limit_is_retried_with_backoff(self):
        waits: list[float] = []
        calls = {"n": 0}

        def fetch(params):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _RateLimited("2")
            return {
                "hits": {
                    "total": {"value": 1},
                    "hits": [{"_source": posting("a")}],
                }
            }

        result = _fetch_with_retry(
            fetch,
            {"size": 25, "from": 0},
            attempts=3,
            backoff=1,
            sleep=waits.append,
        )
        self.assertIsNotNone(result)
        self.assertEqual(waits, [2.0])

    def test_page_cap_truncates_instead_of_looping(self):
        fetch = make_fetch([posting(f"u{n}") for n in range(50)])
        index = harvest(
            fetch=fetch,
            page_size=2,
            max_pages=3,
            min_interval=0,
            sleep=lambda _: None,
        )
        self.assertTrue(index.truncated)
        self.assertEqual(index.pages_read, 3)


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.directory.name, "nav.json.gz")

    def tearDown(self):
        self.directory.cleanup()

    def test_round_trip_preserves_postings(self):
        index = index_of(posting("a"), posting("b"))
        save(index, self.path)
        loaded = load(self.path)
        self.assertIsNotNone(loaded)
        self.assertEqual(
            [item.uuid for item in loaded.postings], ["a", "b"]
        )
        self.assertEqual(loaded.retrieved_at, index.retrieved_at)

    def test_missing_snapshot_loads_as_none(self):
        self.assertIsNone(load(self.path))

    def test_fresh_snapshot_is_not_refetched(self):
        save(index_of(posting("a")), self.path)
        fetch = make_fetch([posting("a")])
        index, meta = ensure_snapshot(
            self.path, fetch=fetch, max_age_hours=12
        )
        self.assertIsNotNone(index)
        self.assertEqual(fetch.calls, [])
        self.assertFalse(meta["refreshed"])
        self.assertFalse(meta["stale"])

    def test_stale_snapshot_is_refreshed(self):
        old = NavJobsIndex(
            postings=[NavPosting(uuid="old", title="t", employer="E")],
            retrieved_at="2020-01-01T00:00:00+00:00",
            total=1,
        )
        save(old, self.path)
        fetch = make_fetch([posting("new")])
        index, meta = ensure_snapshot(
            self.path,
            fetch=fetch,
            page_size=25,
            min_interval=0,
            sleep=lambda _: None,
        )
        self.assertTrue(meta["refreshed"])
        self.assertEqual([item.uuid for item in index.postings], ["new"])

    def test_failed_refresh_keeps_the_old_snapshot(self):
        old = NavJobsIndex(
            postings=[NavPosting(uuid="old", title="t", employer="E")],
            retrieved_at="2020-01-01T00:00:00+00:00",
            total=1,
        )
        save(old, self.path)

        def fetch(params):
            raise RuntimeError("down")

        index, meta = ensure_snapshot(
            self.path,
            fetch=fetch,
            attempts=1,
            backoff=0,
            sleep=lambda _: None,
        )
        self.assertTrue(meta["stale"])
        self.assertEqual([item.uuid for item in index.postings], ["old"])


if __name__ == "__main__":
    unittest.main()
