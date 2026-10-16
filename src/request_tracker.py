"""Outbound request counting and ceiling enforcement.

The limit defaults to the competition's stated daily ceiling of 2,000
and is overridable with ``AGENT_REQUEST_LIMIT``.

That override has to be read *here*. It previously existed only in
``src/budget.py``, while the tracker that actually counts and raises
hardcoded 2,000 -- so setting the variable for a larger build looked
like it worked and silently changed nothing. The two read the same
source now, or the ceiling a run believes it has is not the one it has.
"""

import os

# The competition's stated daily ceiling.
DEFAULT_LIMIT = 2000


def default_limit() -> int:
    """The configured ceiling, falling back to the daily default."""
    raw = (os.getenv("AGENT_REQUEST_LIMIT") or "").strip()
    if not raw:
        return DEFAULT_LIMIT

    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_LIMIT

    return value if value > 0 else DEFAULT_LIMIT


class RequestTracker:
    def __init__(self, limit: int | None = None):
        self.limit = limit if limit is not None else default_limit()
        self.count = 0

    def record(self, requests: int = 1):
        if self.count + requests > self.limit:
            raise RuntimeError(
                f"Outbound request limit exceeded: "
                f"{self.limit}"
            )

        self.count += requests

    def remaining(self) -> int:
        return self.limit - self.count

    def summary(self) -> dict:
        return {
            "requests_used": self.count,
            "request_limit": self.limit,
            "requests_remaining": self.remaining(),
        }


request_tracker = RequestTracker()
