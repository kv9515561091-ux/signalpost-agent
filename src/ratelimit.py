"""
Client-side rate limiting.

NVIDIA's hosted endpoint enforces a request-per-minute ceiling that it
does not publish. Community reports consistently place the default near
40 RPM, and a 429 arrives with no ``Retry-After`` header, so the client
cannot react intelligently to being throttled — it can only wait and try
again, which wastes the whole call.

Pacing requests up front is cheaper and simpler than discovering the
limit by hitting it. At the default of 30 RPM a ten-company batch at the
current extraction depth makes roughly sixty model calls, which a burst
would otherwise exceed on its own.

The limiter is per-process and deliberately simple. It exists to keep a
normal run under the ceiling, not to negotiate with a shared quota.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field

# Kept below the observed default so a run has headroom rather than
# running exactly at the edge of the limit.
DEFAULT_RPM = 30
MIN_RPM = 1
MAX_RPM = 600


def requests_per_minute() -> int:
    """The configured ceiling, from ``NVIDIA_RPM`` or the default."""
    raw = (os.getenv("NVIDIA_RPM") or "").strip()
    if not raw:
        return DEFAULT_RPM

    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_RPM

    return max(MIN_RPM, min(MAX_RPM, value))


@dataclass
class RateLimiter:
    """Enforces a minimum interval between operations.

    Args:
        rpm: Operations permitted per minute. ``0`` or ``None``
            disables limiting entirely.
        clock: Injectable time source, so tests need not sleep.
        sleeper: Injectable sleep, for the same reason.
    """

    rpm: int | None = None
    clock: object = time.monotonic
    sleeper: object = time.sleep
    _last_call: float | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    _waited: float = field(default=0.0, init=False, repr=False)

    @property
    def min_interval(self) -> float:
        """Seconds that must elapse between calls."""
        rpm = self.rpm if self.rpm is not None else requests_per_minute()
        if rpm <= 0:
            return 0.0
        return 60.0 / rpm

    @property
    def total_waited(self) -> float:
        """Seconds spent waiting, for reporting."""
        return self._waited

    def acquire(self) -> float:
        """Block until the next call may proceed. Returns seconds waited."""
        interval = self.min_interval
        if interval <= 0:
            return 0.0

        with self._lock:
            now = self.clock()  # type: ignore[operator]
            if self._last_call is None:
                self._last_call = now
                return 0.0

            earliest = self._last_call + interval
            wait = earliest - now

            if wait > 0:
                self._waited += wait
                self.sleeper(wait)  # type: ignore[operator]
                now = self.clock()  # type: ignore[operator]

            self._last_call = now
            return max(0.0, wait)

    def reset(self) -> None:
        with self._lock:
            self._last_call = None
            self._waited = 0.0


# Process-wide limiter shared by every backend instance, so two backends
# cannot together exceed the ceiling.
limiter = RateLimiter()
