"""Setup diagnostics.

`--check` exists so nobody has to type Python into a shell to find out
whether their setup works. If it is wrong, it is worse than useless: it
sends someone editing code when their problem is a missing key.
"""

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from src.__main__ import main


def run(*args):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = main(list(args))
    return code, buffer.getvalue()


class CheckTests(unittest.TestCase):
    def test_check_requires_no_organization_number(self):
        code, out = run("--check")
        self.assertIn("SETUP CHECK", out)
        self.assertIn(code, (0, 1))

    def test_check_confirms_the_registry_spine_path_always_works(self):
        # Even a fully unconfigured machine must be told what it can do.
        _, out = run("--check")
        self.assertIn("python -m src.agent 989061593", out)

    def test_check_reports_the_active_provider(self):
        with patch("src.llm.provider_name", return_value="openai_compatible"):
            with patch(
                "src.llm.describe",
                return_value="openai_compatible:nvidia/test (thinking off)",
            ):
                _, out = run("--check")
        self.assertIn("openai_compatible:nvidia/test", out)

    def test_check_explains_a_missing_key(self):
        import os

        with patch.dict(os.environ, {}, clear=True):
            with patch("src.llm.provider_name", return_value=None):
                with patch(
                    "src.llm.unavailable_reason",
                    return_value="No LLM configured. Set NVIDIA_API_KEY in .env",
                ):
                    _, out = run("--check")
        self.assertIn("NVIDIA_API_KEY", out)

    def test_check_never_leaks_key_material(self):
        import os

        _, out = run("--check")
        for name in ("TINYFISH_API_KEY", "NVIDIA_API_KEY", "ANTHROPIC_API_KEY"):
            value = os.getenv(name)
            if value and len(value) > 8:
                self.assertNotIn(value, out)
        # The report names the variables, never their values.
        self.assertNotIn("sk-ant-", out)
        self.assertNotIn("nvapi-", out)


class ParityTests(unittest.TestCase):
    """The two entry points must not drift into separate vocabularies.

    The README claimed they took the same flags. They did not: the single
    CLI had no depth flags at all, so ``--max-sources`` meant something
    in one entry point and nothing in the other. This asserts the shared
    half stays shared.
    """

    # Flags that control enrichment, and therefore cost money. These must
    # exist, identically, in both entry points.
    SHARED = (
        "--web",
        "--extract",
        "--narrate",
        "--no-llm",
        "--max-sources",
        "--max-pages",
        "--max-crawl-links",
    )

    @staticmethod
    def _flags(parser):
        return {
            option
            for action in parser._actions
            for option in action.option_strings
        }

    def test_shared_flags_exist_in_both(self):
        import run_batch
        from src.__main__ import build_parser as single

        single_flags = self._flags(single())
        batch_flags = self._flags(run_batch.build_parser())

        for flag in self.SHARED:
            self.assertIn(flag, single_flags, f"{flag} missing from single CLI")
            self.assertIn(flag, batch_flags, f"{flag} missing from batch CLI")

    def test_shared_defaults_agree(self):
        import run_batch
        from src.__main__ import build_parser as single

        single_args = single().parse_args([])
        batch_args = run_batch.build_parser().parse_args([])

        for field in ("max_sources", "max_pages", "max_crawl_links"):
            self.assertEqual(
                getattr(single_args, field),
                getattr(batch_args, field),
                f"{field} defaults disagree",
            )

    def test_batch_only_flags_are_deliberate(self):
        # Recorded so a new batch-only flag is a conscious choice rather
        # than an accident.
        import run_batch
        from src.__main__ import build_parser as single

        self.assertEqual(
            self._flags(run_batch.build_parser()) - self._flags(single()),
            {"--input", "--output", "--delay", "--resume", "--workers"},
        )

    def test_single_only_flags_are_deliberate(self):
        import run_batch
        from src.__main__ import build_parser as single

        self.assertEqual(
            self._flags(single()) - self._flags(run_batch.build_parser()),
            {"--check", "--json", "-v", "--verbose", "--clear-cache"},
        )

    def test_max_crawl_links_zero_is_accepted(self):
        import run_batch
        from src.__main__ import build_parser as single

        self.assertEqual(
            single().parse_args(["--max-crawl-links", "0"]).max_crawl_links, 0
        )
        self.assertEqual(
            run_batch.build_parser()
            .parse_args(["--max-crawl-links", "0"])
            .max_crawl_links,
            0,
        )


class DailyCommandTests(unittest.TestCase):
    """The organizer runs one command against their own company file."""

    def test_daily_parser_accepts_a_file_and_keyless_mode(self):
        import run_daily

        args = run_daily.build_parser().parse_args(
            ["--input", "companies.txt", "--output", "out.json", "--keyless"]
        )
        self.assertEqual(args.input, "companies.txt")
        self.assertEqual(args.output, "out.json")
        self.assertTrue(args.keyless)

    def test_daily_help_does_not_require_a_file(self):
        import run_daily

        with self.assertRaises(SystemExit) as ctx:
            run_daily.build_parser().parse_args(["--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_daily_defaults_are_the_competition_paths(self):
        import os

        import run_daily

        with patch.dict(os.environ, {}, clear=True):
            args = run_daily.build_parser().parse_args([])
        self.assertEqual(args.input, run_daily.COMPETITION_INPUT)
        self.assertEqual(args.output, run_daily.COMPETITION_OUTPUT)
        self.assertFalse(args.keyless)


class StreamingOutputTests(unittest.TestCase):
    """Progress must be visible when stdout is redirected.

    Python block-buffers stdout when it is not a terminal, so a
    redirected batch wrote nothing at all until it finished. A
    fifty-company enriched run takes minutes, and the per-company
    progress line is the whole point -- it was invisible.
    """

    def test_single_cli_enables_line_buffering(self):
        import sys

        from src import __main__ as cli

        calls = {}

        class FakeStream:
            def reconfigure(self, **kwargs):
                calls.update(kwargs)

        with patch.object(sys, "stdout", FakeStream()):
            cli._enable_streaming_output()

        self.assertTrue(calls.get("line_buffering"))

    def test_line_buffering_keeps_utf8(self):
        # Reconfiguring resets the encoding, so the Norwegian text that
        # _force_utf8_output() guarantees has to be reapplied.
        import sys

        from src import __main__ as cli

        calls = {}

        class FakeStream:
            def reconfigure(self, **kwargs):
                calls.update(kwargs)

        with patch.object(sys, "stdout", FakeStream()):
            cli._enable_streaming_output()

        self.assertEqual(calls.get("encoding"), "utf-8")

    def test_a_stream_without_reconfigure_does_not_raise(self):
        import sys

        from src import __main__ as cli

        class OldStream:
            pass

        with patch.object(sys, "stdout", OldStream()):
            cli._enable_streaming_output()  # must not raise

    def test_batch_cli_enables_line_buffering(self):
        import sys

        from src import batch_cli

        calls = {}

        class FakeStream:
            def reconfigure(self, **kwargs):
                calls.update(kwargs)

        with patch.object(sys, "stdout", FakeStream()), patch.object(
            batch_cli, "build_parser"
        ) as parser, self.assertRaises(SystemExit):
            parser.return_value.parse_args.side_effect = SystemExit(0)
            batch_cli.main([])

        self.assertTrue(calls.get("line_buffering"))


class ArgumentTests(unittest.TestCase):
    def test_missing_number_is_an_error_not_a_traceback(self):
        with self.assertRaises(SystemExit) as ctx:
            run()
        self.assertNotEqual(ctx.exception.code, 0)

    def test_clear_cache_needs_no_number(self):
        with patch("src.__main__.cache") as mock_cache:
            mock_cache.clear.return_value = 3
            code, out = run("--clear-cache")
        self.assertEqual(code, 0)
        self.assertIn("Removed 3", out)

    def test_help_lists_the_diagnostic_flag(self):
        with self.assertRaises(SystemExit):
            run("--help")
        # argparse writes to stdout; capture it to be sure the flag is there.
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            with self.assertRaises(SystemExit):
                main(["--help"])
        self.assertIn("--check", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
