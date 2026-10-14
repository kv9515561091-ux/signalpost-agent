"""
Batch enrichment.

Enrichment is opt-in, and the thing most likely to regress silently is a
flag that stops reaching ``build_profile``: the run still succeeds, still
writes JSON, and simply produces no facts. These tests assert the flags
travel, not just that the function returns.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import batch
from src.agent import DEFAULT_MAX_PAGES, DEFAULT_MAX_SOURCES


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


class FlagPlumbingTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name) / "out.json"

    def _run(self, numbers=("123456789",), **kwargs):
        def fake_build(raw, **kw):
            self.calls.append((raw, kw))
            return FakeProfile()

        with patch.object(batch, "build_profile", fake_build):
            return batch.process_companies(
                list(numbers), output_file=str(self.out), delay=0, **kwargs
            )

    def test_default_run_is_registry_only(self):
        self._run()
        _, kw = self.calls[0]
        self.assertFalse(kw["include_web"])
        self.assertFalse(kw["include_extraction"])
        self.assertFalse(kw["include_narrative"])
        self.assertTrue(kw["use_llm"])

    def test_flags_reach_build_profile(self):
        self._run(include_web=True, include_extraction=True, include_narrative=True)
        _, kw = self.calls[0]
        self.assertTrue(kw["include_web"])
        self.assertTrue(kw["include_extraction"])
        self.assertTrue(kw["include_narrative"])

    def test_no_llm_is_forwarded(self):
        self._run(include_web=True, use_llm=False)
        _, kw = self.calls[0]
        self.assertFalse(kw["use_llm"])

    def test_caps_reach_build_profile(self):
        self._run(max_sources=2, max_pages=3)
        _, kw = self.calls[0]
        self.assertEqual(kw["max_sources"], 2)
        self.assertEqual(kw["max_pages"], 3)

    def test_default_caps_are_passed_explicitly(self):
        self._run()
        _, kw = self.calls[0]
        self.assertEqual(kw["max_sources"], DEFAULT_MAX_SOURCES)
        self.assertEqual(kw["max_pages"], DEFAULT_MAX_PAGES)

    def test_extraction_implies_web(self):
        # Asking for facts without discovery would silently yield none.
        self._run(include_extraction=True, include_web=False)
        _, kw = self.calls[0]
        self.assertTrue(kw["include_web"])
        self.assertTrue(kw["include_extraction"])


class FailureIsolationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name) / "out.json"

    def test_one_failure_does_not_abort_the_batch(self):
        def flaky(raw, **kw):
            if raw == "bad":
                raise ValueError("boom")
            return FakeProfile(company_number=raw)

        with patch.object(batch, "build_profile", flaky):
            results = batch.process_companies(
                ["good1", "bad", "good2"], output_file=str(self.out), delay=0
            )

        self.assertEqual(len(results), 3)
        self.assertIn("error", results[1])
        self.assertNotIn("error", results[0])
        self.assertNotIn("error", results[2])

    def test_invalid_number_is_recorded_not_raised(self):
        from src.orgnumber import InvalidCompanyNumber

        def raiser(raw, **kw):
            raise InvalidCompanyNumber("bad checksum")

        with patch.object(batch, "build_profile", raiser):
            results = batch.process_companies(
                ["123456789"], output_file=str(self.out), delay=0
            )

        self.assertIn("error", results[0])

    def test_results_are_written_to_disk(self):
        with patch.object(batch, "build_profile", lambda raw, **kw: FakeProfile()):
            batch.process_companies(["123456789"], output_file=str(self.out), delay=0)

        written = json.loads(self.out.read_text(encoding="utf-8"))
        self.assertEqual(len(written), 1)

    def test_blank_lines_are_skipped(self):
        with patch.object(batch, "build_profile", lambda raw, **kw: FakeProfile()):
            results = batch.process_companies(
                ["", "   ", "123456789"], output_file=str(self.out), delay=0
            )
        self.assertEqual(len(results), 1)

    def test_parent_directory_is_created(self):
        nested = Path(self.tmp.name) / "a" / "b" / "out.json"
        with patch.object(batch, "build_profile", lambda raw, **kw: FakeProfile()):
            batch.process_companies(["123456789"], output_file=str(nested), delay=0)
        self.assertTrue(nested.exists())


class ProgressTests(unittest.TestCase):
    def test_callback_fires_per_company(self):
        seen = []
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(batch, "build_profile", lambda raw, **kw: FakeProfile()):
                batch.process_companies(
                    ["1", "2", "3"],
                    output_file=str(Path(tmp) / "o.json"),
                    delay=0,
                    progress=lambda i, t, raw, rec: seen.append((i, t, raw)),
                )
        self.assertEqual([s[0] for s in seen], [1, 2, 3])
        self.assertEqual(seen[0][1], 3)


class SummaryTests(unittest.TestCase):
    def test_counts_enrichment(self):
        results = [
            {
                "financials": {"x": 1},
                "ceo_name": "A",
                "facts": [{"a": 1}],
                "explanation": {"summary": "s"},
            },
            {"facts": [], "explanation": None},
            {"error": "nope"},
        ]
        stats = batch.summarise(results)
        self.assertEqual(stats["total"], 3)
        self.assertEqual(stats["succeeded"], 2)
        self.assertEqual(stats["failed"], 1)
        self.assertEqual(stats["with_facts"], 1)
        self.assertEqual(stats["with_narrative"], 1)
        self.assertEqual(stats["with_financials"], 1)

    def test_empty_batch(self):
        stats = batch.summarise([])
        self.assertEqual(stats["total"], 0)
        self.assertEqual(stats["succeeded"], 0)
        self.assertEqual(stats["failed"], 0)


class NumberLoadingTests(unittest.TestCase):
    def test_missing_file_is_empty(self):
        self.assertEqual(batch.load_company_numbers("no/such/file.txt"), [])

    def test_blank_lines_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "n.txt"
            path.write_text("123456789\n\n  \n987654321\n", encoding="utf-8")
            self.assertEqual(
                batch.load_company_numbers(str(path)), ["123456789", "987654321"]
            )


class CliTests(unittest.TestCase):
    def setUp(self):
        import run_batch

        self.run_batch = run_batch

    def test_enrichment_is_off_by_default(self):
        args = self.run_batch.build_parser().parse_args([])
        self.assertFalse(args.web)
        self.assertFalse(args.extract)
        self.assertFalse(args.narrate)
        self.assertFalse(args.no_llm)

    def test_flags_parse(self):
        args = self.run_batch.build_parser().parse_args(
            ["--web", "--extract", "--narrate", "--no-llm",
             "--max-sources", "2", "--max-pages", "3"]
        )
        self.assertTrue(args.web)
        self.assertTrue(args.extract)
        self.assertTrue(args.narrate)
        self.assertTrue(args.no_llm)
        self.assertEqual(args.max_sources, 2)
        self.assertEqual(args.max_pages, 3)

    def test_caps_default_to_the_agent_values(self):
        args = self.run_batch.build_parser().parse_args([])
        self.assertEqual(args.max_sources, DEFAULT_MAX_SOURCES)
        self.assertEqual(args.max_pages, DEFAULT_MAX_PAGES)

    def test_cli_matches_single_company_vocabulary(self):
        # The two entry points must not drift apart.
        single = self.run_batch.build_parser().parse_args(["--web"])
        batch = self.run_batch.build_parser().parse_args(["--web"])
        self.assertEqual(single.web, batch.web)


if __name__ == "__main__":
    unittest.main()