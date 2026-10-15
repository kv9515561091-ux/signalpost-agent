"""
Environment loading.

``src/config.py`` locates ``.env`` deliberately rather than letting
python-dotenv search upward from the calling module. In the checkout the
upward walk happened to work -- ``src/`` sits below the repository root,
so ``find_dotenv()`` reached the ``.env`` beside it. It stops working the
moment the package is installed for real, because ``src/`` then lives in
site-packages and the walk can never reach a user's project directory.

``pyproject.toml`` is what made that layout reachable, so this is a
regression the packaging work introduced rather than a bug that predates
it. Locating the file explicitly behaves identically from a checkout and
from an install.

The docstring in an earlier draft of this file claimed the original code
was silently failing in the repository. It was not, and the test that
appeared to prove it was calling from a temp directory instead of from a
module in ``src/``. The tests below reproduce the case that is actually
broken -- an installed location -- rather than a convenient one.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import config

REPO_ROOT = Path(__file__).resolve().parents[1]

_MANAGED = ("SIGNALPOST_", "TINYFISH_", "NVIDIA_", "ANTHROPIC_", "LLM_PROVIDER")


class EnvIsolationMixin(unittest.TestCase):
    """A temp working directory and a cleaned environment per test."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

        self._prev_cwd = Path.cwd()
        os.chdir(self.dir)
        self.addCleanup(os.chdir, self._prev_cwd)

        self._saved = {
            k: v
            for k, v in os.environ.items()
            if k.startswith(_MANAGED)
        }
        for key in self._saved:
            del os.environ[key]

        def restore():
            for key in list(os.environ):
                if key.startswith(_MANAGED):
                    del os.environ[key]
            os.environ.update(self._saved)
            config._loaded = False

        self.addCleanup(restore)
        config._loaded = False

    def write_env(self, body: str, name: str = ".env") -> Path:
        path = self.dir / name
        path.write_text(body, encoding="utf-8")
        return path


class DiscoveryTests(EnvIsolationMixin):
    def test_finds_env_in_the_working_directory(self):
        path = self.write_env("SIGNALPOST_TEST_KEY=from-the-file\n")
        self.assertEqual(config.find_env_file().resolve(), path.resolve())

    def test_finds_env_in_a_parent_directory(self):
        self.write_env("SIGNALPOST_TEST_KEY=from-the-file\n")
        nested = self.dir / "a" / "b"
        nested.mkdir(parents=True)
        os.chdir(nested)
        self.assertIsNotNone(config.find_env_file())

    def test_missing_env_is_not_an_error(self):
        # A deployment injecting credentials through the environment is a
        # legitimate setup; demanding a file would break it.
        self.assertIsNone(config.find_env_file())

    def test_explicit_path_wins(self):
        self.write_env("SIGNALPOST_TEST_KEY=ignored\n")
        with tempfile.TemporaryDirectory() as other:
            explicit = Path(other) / "custom.env"
            explicit.write_text(
                "SIGNALPOST_TEST_KEY=explicit\n", encoding="utf-8"
            )
            with patch.dict(os.environ, {"SIGNALPOST_ENV_FILE": str(explicit)}):
                self.assertEqual(config.find_env_file(), explicit)


class LoadingTests(EnvIsolationMixin):
    def test_reads_the_file(self):
        self.write_env("SIGNALPOST_TEST_KEY=from-the-file\n")
        self.assertIsNotNone(config.load_env(force=True))
        self.assertEqual(os.environ["SIGNALPOST_TEST_KEY"], "from-the-file")

    def test_returns_none_when_absent(self):
        self.assertIsNone(config.load_env(force=True))

    def test_environment_overrides_the_file(self):
        # A key exported in the shell is a deliberate override and must
        # keep winning.
        self.write_env("SIGNALPOST_TEST_KEY=from-the-file\n")
        with patch.dict(os.environ, {"SIGNALPOST_TEST_KEY": "from-the-shell"}):
            config.load_env(force=True)
            self.assertEqual(os.environ["SIGNALPOST_TEST_KEY"], "from-the-shell")

    def test_is_idempotent(self):
        self.write_env("SIGNALPOST_TEST_KEY=from-the-file\n")
        config.load_env(force=True)
        os.environ["SIGNALPOST_TEST_KEY"] = "changed-after-load"
        # A second bare call must not re-read the file over the top.
        self.assertIsNone(config.load_env())
        self.assertEqual(os.environ["SIGNALPOST_TEST_KEY"], "changed-after-load")

    def test_malformed_lines_do_not_raise(self):
        self.write_env("NOT_A_PAIR\nSIGNALPOST_TEST_KEY=ok\n")
        self.assertIsNotNone(config.load_env(force=True))
        self.assertEqual(os.environ["SIGNALPOST_TEST_KEY"], "ok")


class InstalledLayoutTests(EnvIsolationMixin):
    """Why the loader was rewritten, stated accurately.

    An earlier draft of this file claimed bare ``load_dotenv()`` failed
    even in the repository. That was wrong, and the test that "proved" it
    was calling from a temp directory rather than from a real module. In
    the checkout, ``find_dotenv()`` walks up from ``src/`` and reaches the
    repository's ``.env``, so the original code worked.

    What genuinely does break is the layout ``pyproject.toml`` introduced:
    after a non-editable ``pip install .``, ``src/`` lives in
    site-packages, so the walk starts there and never reaches the user's
    project directory. This test reproduces *that*.
    """

    def _run_in_subprocess(self, body: str, cwd: Path, site_dir: Path) -> str:
        script = site_dir / "installed_module.py"
        script.write_text(body, encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            # Deliberately no PYTHONPATH: the probe must not be able to
            # reach the repository, exactly as an installed copy cannot.
            env={
                k: v
                for k, v in os.environ.items()
                if not k.startswith(_MANAGED)
            },
            timeout=60,
        )
        return result.stdout.strip()

    def test_bare_load_dotenv_fails_from_an_installed_location(self):
        self.write_env("SIGNALPOST_TEST_KEY=from-the-file\n")

        with tempfile.TemporaryDirectory() as fake_site:
            probe = (
                "import os, sys\n"
                "sys.path.insert(0, os.path.dirname(__file__))\n"
                "from dotenv import load_dotenv\n"
                "print(f'bare_returned={load_dotenv()}')\n"
                "print(f'bare_value={os.environ.get(\"SIGNALPOST_TEST_KEY\")}')\n"
            )
            out = self._run_in_subprocess(
                probe, self.dir, Path(fake_site)
            )

        self.assertIn("bare_returned=False", out, out)
        self.assertIn("bare_value=None", out, out)

    def test_explicit_loader_finds_the_projects_env(self):
        # Same scenario, same cwd, but locating the file deliberately.
        self.write_env("SIGNALPOST_TEST_KEY=from-the-file\n")

        with tempfile.TemporaryDirectory() as fake_site:
            probe = (
                f"import os, sys\n"
                f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
                f"from src.config import load_env\n"
                f"print(f'fixed_returned={{load_env() is not None}}')\n"
                f"print(f'fixed_value={{os.environ.get(\"SIGNALPOST_TEST_KEY\")}}')\n"
            )
            out = self._run_in_subprocess(
                probe, self.dir, Path(fake_site)
            )

        self.assertIn("fixed_returned=True", out, out)
        self.assertIn("fixed_value=from-the-file", out, out)

    def test_modules_no_longer_rely_on_caller_relative_discovery(self):
        import ast

        for module in ("src/llm.py", "src/tinyfish_search.py"):
            tree = ast.parse((REPO_ROOT / module).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = getattr(func, "attr", None) or getattr(func, "id", None)
                if name == "load_dotenv" and not node.args:
                    self.fail(
                        f"{module} calls load_dotenv() with no path, which "
                        "searches from the module's own location and cannot "
                        "reach a user's project once installed; use "
                        "src.config.load_env instead"
                    )


class CredentialTests(EnvIsolationMixin):
    def test_credentials_present_reflects_the_file(self):
        self.write_env("TINYFISH_API_KEY=abc\n")
        self.assertTrue(config.credentials_present())

    def test_no_file_and_no_environment_is_absent(self):
        self.assertFalse(config.credentials_present())

    def test_empty_value_is_not_a_credential(self):
        self.write_env("TINYFISH_API_KEY=\n")
        self.assertFalse(config.credentials_present())

    def test_environment_only_setup_is_supported(self):
        # No .env at all, but a key in the environment: legitimate.
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "nvapi-x"}):
            self.assertTrue(config.credentials_present())

    def test_loading_is_not_repeated_once_done(self):
        # Reading the file once per process is deliberate: a key exported
        # after startup must not be silently overwritten by the file.
        self.write_env("SIGNALPOST_TEST_KEY=first\n")
        config.load_env(force=True)
        os.environ["SIGNALPOST_TEST_KEY"] = "second"
        self.assertIsNone(config.load_env())
        self.assertEqual(os.environ["SIGNALPOST_TEST_KEY"], "second")


if __name__ == "__main__":
    unittest.main()
