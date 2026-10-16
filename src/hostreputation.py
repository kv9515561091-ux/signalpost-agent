"""
Which hosts are worth talking to again.

A batch of a hundred companies meets the same directories over and over.
The fifty-company run measured 95 retrieval failures, 63 of them on five
hosts that refuse automated clients outright: gulesider.no (22),
firmadatabasen.no (21), purehelp.no (11), altlokalt.no (3). Each of
those was a request spent to learn the same thing again.

The refusal is stable rather than incidental, which is what makes it
worth remembering. A site that answers 403 to a crawler will answer 403
to the next one too, so the second refusal is enough to stop asking for
the rest of the run.

Only definitive refusals are remembered. A timeout or a TLS failure says
something about one fetch, not about the host, and a 404 says something
about one path -- neither should cost the other pages on the same site.
"""

from __future__ import annotations

from urllib.parse import urlparse

# A host is abandoned after this many refusals. Two, not one: a single
# 429 is often a momentary blip, and giving up immediately would lose
# real pages for the sake of saving one request.
STRIKES_TO_BLOCK = 2

# Response statuses that mean "this host will not serve us".
# 403 is the common case for Norwegian directories behind a bot wall;
# 401 is the same idea with a login; 429 means we are being throttled
# and hammering it makes that worse.
REFUSAL_STATUSES = frozenset({401, 403, 429})


class HostReputation:
    """Per-run record of which hosts refuse automated access."""

    def __init__(self, strikes_to_block: int = STRIKES_TO_BLOCK):
        self.strikes: dict[str, int] = {}
        self.blocked: set[str] = set()
        self.strikes_to_block = strikes_to_block
        # Requests we did not make because the host was already blocked.
        self.saved = 0

    @staticmethod
    def host_of(url: str) -> str:
        return (urlparse(url).netloc or "").lower()

    def record(self, url: str, status_code: int) -> None:
        """Note a refusal and block the host once it persists."""
        if status_code not in REFUSAL_STATUSES:
            return

        host = self.host_of(url)
        if not host:
            return

        count = self.strikes.get(host, 0) + 1
        self.strikes[host] = count

        if count >= self.strikes_to_block:
            self.blocked.add(host)

    def is_blocked(self, url_or_host: str) -> bool:
        host = (
            url_or_host
            if "://" not in url_or_host
            else self.host_of(url_or_host)
        )
        return host in self.blocked

    def should_skip(self, url: str) -> bool:
        """Whether to spend a request on this URL at all."""
        if self.is_blocked(url):
            self.saved += 1
            return True
        return False

    def note_skipped(self, url: str) -> None:
        """Record a skip decision explicitly."""
        self.saved += 1
        _ = url

    def summary(self) -> dict:
        return {
            "hosts_blocked": len(self.blocked),
            "requests_avoided": self.saved,
            "blocked": sorted(self.blocked),
        }

    def reset(self) -> None:
        self.strikes.clear()
        self.blocked.clear()
        self.saved = 0


# One per process: a batch is the unit over which a refusal is stable.
reputation = HostReputation()
