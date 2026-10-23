"""
Concurrency invariants.

The competition ceiling is 2,000 outbound requests and exceeding it
disqualifies the entry, so the counter has to stay exact when companies
are profiled in parallel -- a lost increment is not a slow run, it is a
failed one. The rest of these assert that parallelism changes only the
order work happens in, never the answer: every company still gets a
profile, failures stay isolated, and the output file keeps input order.
"""

import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from src import batch
from src.hostreputation import HostReputation
from src.request_tracker import RequestTracker


class FakeProfile:
    def __init__(self, **overrides):
        self._data = {
            "company_number": "123456789",
            "company_name": "EXAMPLE AS",
            "facts": [],
            "explanation": None,
            "financials": None,
            "ceo_name": None,
        }
        self._data.update(overrides)

    def model_dump(self, mode="json"):
        return dict(self._data)


class TrackerUnderConcurrencyTests(unittest.TestCase):
    def test_no_increment_is_lost(self):
        tracker = RequestTracker(10_000)

        def worker():
            for _ in range(500):
                tracker.record()

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(tracker.count, 4_000)

    def test_ceiling_holds_under_contention(self):
        tracker = RequestTracker(500)
        refusals = []
        started = threading.Barrier(8)

        def worker():
            started.wait()
            for _ in range(200):
                try:
                    tracker.record()
                except RuntimeError:
                    refusals.append(1)
                    return

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Exactly the ceiling, never one over -- and the losers stopped.
        self.assertEqual(tracker.count, 500)
        self.assertTrue(refusals)


class ReputationUnderConcurrencyTests(unittest.TestCase):
    def test_concurrent_refusals_still_block_the_host(self):
        rep = HostReputation()

        def worker():
            for _ in range(50):
                rep.record("https://gulesider.no/company/1", 403)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertTrue(rep.is_blocked("gulesider.no"))
        self.assertEqual(rep.strikes["gulesider.no"], 400)

    def test_skip_count_is_exact(self):
        rep = HostReputation()
        rep.blocked.add("gulesider.no")

        def worker():
            for _ in range(100):
                rep.should_skip("https://gulesider.no/x")

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(rep.saved, 400)


class ParallelBatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name) / "out.json"

    def test_every_company_is_profiled_and_output_keeps_input_order(self):
        numbers = [f"{i:09d}" for i in range(1, 21)]

        def fake_build(raw, **kwargs):
            # Finishes in reverse order, which is the point: the output
            # order must come from the input, not from the clock.
            time.sleep(0.002 * (21 - int(raw)))
            return FakeProfile(company_number=raw)

        with patch.object(batch, "build_profile", fake_build):
            results = batch.process_companies(
                numbers,
                output_file=str(self.out),
                delay=0,
                workers=5,
            )

        self.assertEqual([r["company_number"] for r in results], numbers)
        written = json.loads(self.out.read_text(encoding="utf-8"))
        self.assertEqual([r["company_number"] for r in written], numbers)

    def test_progress_fires_once_per_company(self):
        numbers = ["1", "2", "3", "4", "5"]
        seen = []

        with patch.object(
            batch,
            "build_profile",
            lambda raw, **kw: FakeProfile(company_number=raw),
        ):
            batch.process_companies(
                numbers,
                output_file=str(self.out),
                delay=0,
                workers=3,
                progress=lambda i, t, raw, rec: seen.append((i, raw)),
            )

        self.assertEqual(sorted(seen), [(1, "1"), (2, "2"), (3, "3"),
                                        (4, "4"), (5, "5")])

    def test_one_failure_does_not_abort_a_parallel_run(self):
        numbers = ["111111111", "222222222", "bad", "444444444"]

        def fake_build(raw, **kwargs):
            if raw == "bad":
                raise ValueError("boom")
            return FakeProfile(company_number=raw)

        with patch.object(batch, "build_profile", fake_build):
            results = batch.process_companies(
                numbers, output_file=str(self.out), delay=0, workers=4
            )

        self.assertEqual(len(results), 4)
        self.assertEqual(results[2]["company_number"], "bad")
        self.assertIn("error", results[2])
        self.assertNotIn("error", results[0])
        self.assertNotIn("error", results[3])

    def test_absurd_worker_count_is_clamped_not_fatal(self):
        with patch.object(
            batch,
            "build_profile",
            lambda raw, **kw: FakeProfile(company_number=raw),
        ):
            results = batch.process_companies(
                ["1", "2"], output_file=str(self.out), delay=0, workers=999
            )

        self.assertEqual(len(results), 2)


class WorkerConfigTests(unittest.TestCase):
    def test_env_sets_the_worker_count(self):
        with patch.dict(os.environ, {"AGENT_WORKERS": "6"}):
            self.assertEqual(batch.default_workers(), 6)

    def test_unset_env_returns_the_fallback(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AGENT_WORKERS", None)
            self.assertEqual(batch.default_workers(fallback=8), 8)

    def test_nonsense_env_returns_the_fallback(self):
        with patch.dict(os.environ, {"AGENT_WORKERS": "lots"}):
            self.assertEqual(batch.default_workers(fallback=3), 3)

    def test_zero_is_clamped_to_one(self):
        with patch.dict(os.environ, {"AGENT_WORKERS": "0"}):
            self.assertEqual(batch.default_workers(), 1)


if __name__ == "__main__":
    unittest.main()
