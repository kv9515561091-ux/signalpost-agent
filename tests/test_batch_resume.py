"""
Batch resume.

The results file used to be written once, after the last company. A
fifty-company enriched run is minutes of billable calls, and anything
that went wrong at number thirty -- an exception escaping the per-company
guard, a laptop lid closed, a rate limit -- discarded all thirty finished
profiles. The per-company error handling made a single bad company
survivable and made a bad *run* unsurvivable, which is backwards.

Results are now written after every company, through a temporary file and
an atomic rename, so an interrupted run keeps what it completed.

Resume refuses to reuse results produced under different settings. A
registry-only profile is a valid answer to a different question than an
enriched one, and silently mixing the two into one output file is worse
than redoing paid work.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import batch


def _interrupt():
    raise KeyboardInterrupt("process died here")


class FakeProfile:
    def __init__(self, number: str, **overrides):
        self.facts = []
        self._data = {
            "company_number": number,
            "company_name": f"COMPANY {number} AS",
            "facts": [],
            "explanation": None,
            "financials": None,
            "ceo_name": None,
        }
        self._data.update(overrides)

    def model_dump(self, mode="json"):
        return dict(self._data)


class BatchFixture(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.out = str(self.dir / "out.json")
        self.calls: list[str] = []

    def _patch(self, behaviour=None):
        def fake_build(raw, **kwargs):
            self.calls.append(raw)
            if behaviour is not None:
                outcome = behaviour(raw)
                if isinstance(outcome, Exception):
                    raise outcome
                if isinstance(outcome, dict):
                    return FakeProfile(raw, **outcome)
            return FakeProfile(raw)

        patcher = patch.object(batch, "build_profile", fake_build)
        patcher.start()
        self.addCleanup(patcher.stop)


class CrashRecoveryTests(BatchFixture):
    def test_results_survive_an_interrupted_run(self):
        """The bug. A crash must not discard completed work.

        KeyboardInterrupt rather than an ordinary exception: the
        per-company guard catches Exception on purpose, so a raised
        error becomes a record rather than ending the run. Only something
        escaping Exception models a process that actually died.
        """
        self._patch()

        with patch.object(batch.time, "sleep", lambda _s: None):
            with self.assertRaises(KeyboardInterrupt):
                batch.process_companies(
                    ["1", "2", "3"],
                    output_file=self.out,
                    delay=0,
                    progress=lambda i, t, raw, rec: (
                        _interrupt()
                        if raw == "3"
                        else None
                    ),
                )

        # Everything completed before the crash is on disk.
        saved = json.loads(Path(self.out).read_text(encoding="utf-8"))
        self.assertEqual([r["company_number"] for r in saved], ["1", "2", "3"])

    def test_output_is_written_after_each_company(self):
        self._patch()
        with patch.object(batch.time, "sleep", lambda _s: None):
            batch.process_companies(["1", "2"], output_file=self.out, delay=0)

        self.assertTrue(Path(self.out).is_file())
        self.assertEqual(len(json.loads(Path(self.out).read_text(encoding="utf-8"))), 2)

    def test_a_partial_write_never_corrupts_the_file(self):
        # Rewriting after every company means a crash during the write
        # itself must not leave truncated JSON.
        self._patch()
        with patch.object(batch.time, "sleep", lambda _s: None):
            batch.process_companies(["1"], output_file=self.out, delay=0)

        good = Path(self.out).read_text(encoding="utf-8")

        real_replace = batch.os.replace

        def exploding_replace(src, dst):
            raise OSError("disk full")

        with patch.object(batch.os, "replace", exploding_replace):
            with self.assertRaises(OSError):
                batch.process_companies(
                    ["1", "2"], output_file=self.out, delay=0
                )

        self.assertEqual(
            Path(self.out).read_text(encoding="utf-8"), good,
            "a failed write must leave the previous complete file intact",
        )
        self.assertTrue(callable(real_replace))

    def test_no_temporary_files_are_left_behind(self):
        self._patch()
        with patch.object(batch.time, "sleep", lambda _s: None):
            batch.process_companies(["1", "2"], output_file=self.out, delay=0)

        leftovers = [p.name for p in self.dir.glob("*.tmp")]
        self.assertEqual(leftovers, [], "temporary files were not cleaned up")


class ResumeTests(BatchFixture):
    def _run(self, numbers, **kwargs):
        with patch.object(batch.time, "sleep", lambda _s: None):
            return batch.process_companies(
                numbers, output_file=self.out, delay=0, **kwargs
            )

    def test_completed_companies_are_skipped(self):
        self._patch()
        self._run(["1", "2", "3"])
        self.assertEqual(self.calls, ["1", "2", "3"])

        self.calls.clear()
        results = self._run(["1", "2", "3"], resume=True)

        self.assertEqual(self.calls, [], "resume re-ran everything")
        self.assertEqual(len(results), 3, "previous results were lost")

    def test_a_new_company_is_still_processed(self):
        self._patch()
        self._run(["1", "2"])
        self.calls.clear()

        self._run(["1", "2", "3"], resume=True)

        self.assertEqual(self.calls, ["3"])

    def test_a_resumed_run_drops_companies_not_in_the_current_input(self):
        self._patch()
        self._run(["1", "2"])
        self.calls.clear()

        results = self._run(["1"], resume=True)

        self.assertEqual(self.calls, [])
        self.assertEqual(
            [row["company_number"] for row in results], ["1"]
        )
        state = json.loads(
            Path(f"{self.out}{batch.STATE_SUFFIX}").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(sorted(state["completed"]), ["1"])

    def test_errored_companies_are_retried(self):
        """A transient failure must not become permanent."""

        attempts = {"n": 0}

        def behaviour(raw):
            if raw == "flaky":
                attempts["n"] += 1
                if attempts["n"] == 1:
                    return ConnectionError("temporary")
            return None

        self._patch(behaviour)

        first = self._run(["ok", "flaky"])
        self.assertEqual(sum(1 for r in first if "error" in r), 1)

        self.calls.clear()
        second = self._run(["ok", "flaky"], resume=True)

        self.assertEqual(self.calls, ["flaky"], "the failed company was not retried")
        self.assertFalse(any("error" in r for r in second))

    def test_resume_is_off_by_default(self):
        self._patch()
        self._run(["1", "2"])
        self.calls.clear()
        self._run(["1", "2"])
        self.assertEqual(self.calls, ["1", "2"], "default run reused old results")

    def test_no_state_file_means_a_fresh_run(self):
        self._patch()
        results = self._run(["1", "2"], resume=True)
        self.assertEqual(self.calls, ["1", "2"])
        self.assertEqual(len(results), 2)


class ModeSafetyTests(BatchFixture):
    """The reason the state file records a signature at all."""

    def _run(self, numbers, **kwargs):
        with patch.object(batch.time, "sleep", lambda _s: None):
            return batch.process_companies(
                numbers, output_file=self.out, delay=0, **kwargs
            )

    def test_different_settings_do_not_reuse(self):
        self._patch()
        self._run(["1", "2"])  # registry-only
        self.calls.clear()

        # A registry-only profile is not the answer to an enriched query.
        self._run(["1", "2"], resume=True, include_extraction=True)

        self.assertEqual(
            self.calls,
            ["1", "2"],
            "resume reused registry-only results for an enriched run",
        )

    def test_different_caps_do_not_reuse(self):
        self._patch()
        self._run(["1"], include_extraction=True, max_pages=10)
        self.calls.clear()

        self._run(["1"], resume=True, include_extraction=True, max_pages=3)

        self.assertEqual(self.calls, ["1"], "caps changed but results were reused")

    def test_no_llm_versus_llm_do_not_mix(self):
        self._patch()
        self._run(["1"], use_llm=False)
        self.calls.clear()
        self._run(["1"], resume=True, use_llm=True)
        self.assertEqual(self.calls, ["1"])

    def test_signature_distinguishes_modes(self):
        base = {
            "include_web": False,
            "include_extraction": False,
            "include_narrative": False,
            "use_llm": True,
            "max_sources": 6,
            "max_pages": 10,
            "max_crawl_links": 5,
        }
        registry_only = batch.mode_signature(**base)
        enriched = batch.mode_signature(**{**base, "include_extraction": True})

        self.assertNotEqual(registry_only, enriched)

    def test_signature_is_stable(self):
        base = {
            "include_web": True,
            "include_extraction": False,
            "include_narrative": False,
            "use_llm": True,
            "max_sources": 6,
            "max_pages": 10,
            "max_crawl_links": 5,
        }
        self.assertEqual(batch.mode_signature(**base), batch.mode_signature(**base))


class CorruptStateTests(BatchFixture):
    def _run(self, numbers, **kwargs):
        with patch.object(batch.time, "sleep", lambda _s: None):
            return batch.process_companies(
                numbers, output_file=self.out, delay=0, **kwargs
            )

    def test_corrupt_state_file_degrades_to_a_full_run(self):
        self._patch()
        Path(f"{self.out}{batch.STATE_SUFFIX}").write_text(
            "{ not json", encoding="utf-8"
        )

        results = self._run(["1", "2"], resume=True)

        self.assertEqual(self.calls, ["1", "2"])
        self.assertEqual(len(results), 2)

    def test_corrupt_results_file_degrades_to_a_full_run(self):
        self._patch()
        Path(self.out).write_text("]]]not json[[[", encoding="utf-8")

        results = self._run(["1"], resume=True)

        self.assertEqual(self.calls, ["1"])
        self.assertEqual(len(results), 1)

    def test_state_without_results_file_is_ignored(self):
        self._patch()
        Path(f"{self.out}{batch.STATE_SUFFIX}").write_text(
            json.dumps({"mode": "registry|s6|p10|c5", "completed": ["1"]}),
            encoding="utf-8",
        )

        self._run(["1"], resume=True)
        self.assertEqual(self.calls, ["1"])


class LoadCompanyNumbersTests(unittest.TestCase):
    """A UTF-8 BOM must not become part of the first org number."""

    def test_bom_is_not_part_of_the_first_number(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "companies.txt"
            path.write_bytes(b"\xef\xbb\xbf916627939\n988539473\n")
            self.assertEqual(
                batch.load_company_numbers(str(path)),
                ["916627939", "988539473"],
            )

    def test_blank_lines_and_whitespace_are_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "companies.txt"
            path.write_bytes(b"  916627939  \n\n988539473\n")
            self.assertEqual(
                batch.load_company_numbers(str(path)),
                ["916627939", "988539473"],
            )


class StateIsCompleteTests(unittest.TestCase):
    """A finished day must refresh; only an interrupted one resumes."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out = str(Path(self._tmp.name) / "out.json")

    def _write(self, completed):
        Path(f"{self.out}{batch.STATE_SUFFIX}").write_text(
            json.dumps({"mode": "x", "completed": completed}),
            encoding="utf-8",
        )

    def test_all_requested_companies_completed(self):
        self._write(["1", "2"])
        self.assertTrue(batch.state_is_complete(self.out, ["1", "2"]))

    def test_one_missing_company_is_not_complete(self):
        self._write(["1"])
        self.assertFalse(batch.state_is_complete(self.out, ["1", "2"]))

    def test_absent_state_is_not_complete(self):
        self.assertFalse(batch.state_is_complete(self.out, ["1"]))

    def test_empty_request_is_not_complete(self):
        self._write([])
        self.assertFalse(batch.state_is_complete(self.out, []))


class StateShapeTests(BatchFixture):
    def test_results_file_remains_a_plain_list(self):
        # Anything consuming this file should not have to unwrap a
        # metadata envelope.
        self._patch()
        with patch.object(batch.time, "sleep", lambda _s: None):
            batch.process_companies(["1"], output_file=self.out, delay=0)

        loaded = json.loads(Path(self.out).read_text(encoding="utf-8"))
        self.assertIsInstance(loaded, list)
        self.assertIsInstance(loaded[0], dict)

    def test_state_file_records_mode_and_completed(self):
        self._patch()
        with patch.object(batch.time, "sleep", lambda _s: None):
            batch.process_companies(["1", "2"], output_file=self.out, delay=0)

        state = json.loads(
            Path(f"{self.out}{batch.STATE_SUFFIX}").read_text(encoding="utf-8")
        )
        self.assertIn("mode", state)
        self.assertEqual(sorted(state["completed"]), ["1", "2"])


if __name__ == "__main__":
    unittest.main()
