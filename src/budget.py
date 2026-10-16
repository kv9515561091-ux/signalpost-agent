"""
Outbound request budget.

The daily limit is hard and exceeding it disqualifies the entry, so the
agent must not merely count requests -- it has to notice it is running
out and stop spending.

Two failure modes matter and they are opposites. A run that stops dead
on the limit throws away the work already done; a run that never notices
until the exception propagates dies mid-company and, in a batch, takes
every remaining company with it. Degradation is the middle path: as the
budget drains, the agent reads fewer pages per company, and past a
floor it falls back to the registry spine alone, which still produces a
valid profile for every company.

The registry is protected. It is free, unauthenticated, three requests
per company, and the backbone of every profile, so it is never the
thing that gets cut.
"""

from __future__ import annotations

import os

# Environment override for the competition's stated daily ceiling.
DEFAULT_LIMIT = int(os.getenv("AGENT_REQUEST_LIMIT", "2000"))

# The registry spine: company record, roles, annual accounts. Free,
# unauthenticated, and the backbone of every profile, so it is the last
# thing that gets cut.
REGISTRY_REQUESTS_PER_COMPANY = 3


class RequestBudget:
    """A ceiling on outbound work, with graceful degradation."""

    def __init__(self, limit: int = DEFAULT_LIMIT, *, protected: int = 0):
        self.limit = limit
        self.used = 0
        # Requests the registry spine needs regardless: three per
        # company, free and unauthenticated.
        self.protected = protected
        # Pages granted to a company but not yet observed by the
        # tracker. Without this the smoothing calculation re-grants the
        # same remainder to every remaining company: ten companies and
        # five requests would allocate zero, zero, zero... then one, one,
        # two, five -- ten in total against five available.
        self.reserved = 0
        self.degraded_from: int | None = None

    @classmethod
    def from_tracker(cls, tracker, *, protected: int = 0) -> RequestBudget:
        """Adopt an existing tracker's limit and current spend."""
        budget = cls(tracker.limit, protected=protected)
        budget.used = tracker.count
        return budget

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used - self.reserved)

    @property
    def fraction_used(self) -> float:
        if self.limit <= 0:
            return 1.0
        return min(1.0, (self.used + self.reserved) / self.limit)

    def spend(self, n: int = 1) -> None:
        self.used += n

    def commit(self, n: int) -> None:
        """Reserve ``n`` requests granted but not yet made.

        The tracker observes a request only once it happens; between
        granting a company its pages and those pages actually going out,
        the budget has to account for both or it over-promises.
        """
        self.reserved = max(0, self.reserved + n)

    def sync(self, tracker) -> None:
        """Adopt the tracker's spend without discarding reservations."""
        self.limit = tracker.limit
        self.used = tracker.count

    def exhausted(self) -> bool:
        return self.remaining <= 0

    def reserve(self, per_company: int, companies_left: int) -> int:
        """The share of the remaining budget one company may spend.

        Smoothing the remainder across the companies still to do is what
        turns "the run died at company 80" into "the last twenty
        companies got shallower but every one of them finished".
        """
        if companies_left <= 0:
            return 0
        return max(0, self.remaining // companies_left)

    def pages_allowed(self, desired: int, per_company: int, companies_left: int) -> int:
        """How many pages this company may read, given what is left.

        Always returns at least one while any budget remains, because a
        company with zero pages read contributes nothing from the web
        and is strictly worse than one page read.
        """
        if per_company <= 0:
            return 0
        allowance = self.reserve(per_company, companies_left)
        if allowance <= 0:
            return 0
        return max(1, min(desired, allowance))

    def registry_only(self) -> bool:
        """Whether to stop enrichment and publish the registry spine."""
        return self.remaining <= self.protected

    def summary(self) -> dict:
        return {
            "budget_limit": self.limit,
            "budget_used": self.used,
            "budget_reserved": self.reserved,
            "budget_remaining": self.remaining,
            "budget_used_pct": round(self.fraction_used * 100, 1),
            "enrichment_disabled_for_budget": self.registry_only(),
        }


# Process-wide, mirroring the request tracker it governs.
budget = RequestBudget()
