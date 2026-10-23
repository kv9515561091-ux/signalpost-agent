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
import threading

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
        # Concurrency turns the check-then-increment below into a race,
        # and a lost increment is how a run exceeds the ceiling it
        # believes it is enforcing. The ceiling is a disqualification
        # boundary, not a target, so it gets a lock rather than a hope
        # that CPython's GIL happens to make it atomic.
        self._lock = threading.Lock()

    def record(self, requests: int = 1):
        with self._lock:
            if self.count + requests > self.limit:
                raise RuntimeError(
                    f"Outbound request limit exceeded: "
                    f"{self.limit}"
                )

            self.count += requests

    def remaining(self) -> int:
        with self._lock:
            return self.limit - self.count

    def summary(self) -> dict:
        # Read under the lock, and compute the remainder here rather than
        # calling ``remaining()``, which would try to take the
        # non-reentrant lock again and deadlock.
        with self._lock:
            used = self.count
            return {
                "requests_used": used,
                "request_limit": self.limit,
                "requests_remaining": self.limit - used,
            }


request_tracker = RequestTracker()
