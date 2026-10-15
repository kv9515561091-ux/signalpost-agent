"""
Environment loading.

`load_dotenv()` with no argument searches upward from the **calling
module's file**, not from the working directory. Both `src/llm.py` and
`src/tinyfish_search.py` called it bare, and it returned ``False`` -- the
repo's `.env` was never opened, and the agent ran on whatever credential
happened to be in the process environment.

That is also a packaging bug rather than merely a clumsy one: after
`pip install .`, ``src/`` lives in ``site-packages``, so the upward walk
starts there and can never reach the user's project directory.

So the file is located deliberately instead. A real environment variable
still wins over the file, which is what a user exporting a key in a shell
expects, and an explicit path always wins over both.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_FILENAME = ".env"

_loaded = False


def _candidate_paths() -> list[Path]:
    """Where a ``.env`` may legitimately live, most specific first.

    An explicit path always wins, so it is checked alone. Otherwise the
    search runs from the working directory outward, which is how a user
    invokes a CLI and how it behaves identically whether the code came
    from a checkout or from site-packages.
    """
    explicit = os.getenv("SIGNALPOST_ENV_FILE")
    if explicit:
        return [Path(explicit).expanduser()]

    candidates: list[Path] = []
    cwd = Path.cwd().resolve()
    candidates.append(cwd / ENV_FILENAME)
    candidates.extend(parent / ENV_FILENAME for parent in cwd.parents)
    return candidates


def find_env_file() -> Path | None:
    """The ``.env`` that would be read, or ``None`` if there is none."""
    for candidate in _candidate_paths():
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def load_env(*, force: bool = False) -> Path | None:
    """Populate ``os.environ`` from the located ``.env``.

    Idempotent: the file is read once per process unless ``force`` is set,
    so importing two modules that both want credentials reads it once.

    Returns the path read, or ``None`` when no ``.env`` exists. Existing
    environment variables are never overwritten -- a key exported in the
    shell is a deliberate override and must keep winning.

    Note:
        This does not raise when the file is missing. A deployment that
        injects credentials purely through the environment is a legitimate
        setup, and demanding a file would break it.
    """
    global _loaded

    if _loaded and not force:
        return None

    path = find_env_file()
    _loaded = True

    if path is None:
        return None

    from dotenv import dotenv_values

    try:
        values = dotenv_values(path)
    except OSError:
        return None

    for key, value in values.items():
        if value is None:
            continue
        if key in os.environ:
            continue
        os.environ[key] = value

    return path


def credentials_present() -> bool:
    """Whether at least one LLM or search credential is visible.

    Kept beside the loader so "did the file actually load" and "is a key
    present" are answerable without guessing which module ran first.
    """
    load_env()
    return any(
        os.getenv(name)
        for name in ("TINYFISH_API_KEY", "ANTHROPIC_API_KEY", "NVIDIA_API_KEY")
    )


__all__ = [
    "ENV_FILENAME",
    "credentials_present",
    "find_env_file",
    "load_env",
]
