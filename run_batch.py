"""Root shim so ``python run_batch.py`` keeps working.

The implementation lives in :mod:`src.batch_cli` so it is importable and
can be shipped as a ``signalpost-batch`` console script. Keeping this
file means the documented invocation in the README stays true.
"""

from src.batch_cli import build_parser, main

__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
