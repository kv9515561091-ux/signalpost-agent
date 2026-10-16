"""
Host reputation.

A batch of a hundred companies meets the same directories repeatedly. The
fifty-company run spent 95 retrieval failures, 63 of them on five hosts
that refuse automated clients outright. Each of those was a request spent
to learn something already known.

The trade-off is recall against waste. Blocking too eagerly loses real
pages; too late and the budget pays for the lesson twice. Two strikes
with block-on-refusal-only is the setting that gives that up.
"""

import contextlib
import unittest

from src.hostreputation import (
    REFUSAL_STATUSES,
    STRIKES_TO_BLOCK,
    HostReputation,
)


class StrikeTests(unittest.TestCase):
    def setUp(self):
        self.rep = HostReputation()

    def test_one_refusal_does_not_block(self):
        # A single 429 is often momentary. Blocking on it would cost
        # real pages to save one request.
        self.rep.record("https://example.no/x", 403)
        self.assertFalse(self.rep.is_blocked("https://example.no/x"))

    def test_two_refusals_block(self):
        for _ in range(STRIKES_TO_BLOCK):
            self.rep.record("https://example.no/x", 403)
        self.assertTrue(self.rep.is_blocked("https://example.no/x"))

    def test_strikes_are_per_host(self):
        for _ in range(STRIKES_TO_BLOCK):
            self.rep.record("https://blocked.no/x", 403)
        self.assertTrue(self.rep.is_blocked("https://blocked.no"))
        self.assertFalse(self.rep.is_blocked("https://innocent.no"))

    def test_non_refusal_statuses_are_not_strikes(self):
        # A 404 is about one path, a 200 is a success, a 500 might be a
        # blip. None say anything about the host.
        for status in (200, 301, 404, 410, 500, 502):
            for _ in range(5):
                self.rep.record("https://example.no/x", status)
            self.assertFalse(
                self.rep.is_blocked("https://example.no"),
                f"{status} should not block a host",
            )

    def test_every_refusal_status_blocks(self):
        for status in REFUSAL_STATUSES:
            rep = HostReputation()
            for _ in range(STRIKES_TO_BLOCK):
                rep.record("https://example.no/x", status)
            self.assertTrue(
                rep.is_blocked("https://example.no"), f"{status} should block"
            )

    def test_https_and_http_share_a_host(self):
        rep = HostReputation()
        rep.record("https://example.no/a", 403)
        rep.record("http://example.no/b", 403)
        self.assertTrue(rep.is_blocked("https://example.no"))

    def test_host_case_is_ignored(self):
        rep = HostReputation()
        for _ in range(STRIKES_TO_BLOCK):
            rep.record("https://EXAMPLE.NO/x", 403)
        self.assertTrue(rep.is_blocked("https://example.no"))


class LookupTests(unittest.TestCase):
    def setUp(self):
        self.rep = HostReputation()
        for _ in range(STRIKES_TO_BLOCK):
            self.rep.record("https://gulesider.no/x", 403)

    def test_accepts_a_bare_host(self):
        self.assertTrue(self.rep.is_blocked("gulesider.no"))

    def test_accepts_a_full_url(self):
        self.assertTrue(self.rep.is_blocked("https://gulesider.no/bedrift/1"))

    def test_should_skip_counts_the_avoided_request(self):
        self.assertTrue(self.rep.should_skip("https://gulesider.no/x"))
        self.assertEqual(self.rep.saved, 1)

    def test_should_skip_false_for_a_useful_host(self):
        self.assertFalse(self.rep.should_skip("https://equinor.com"))
        self.assertEqual(self.rep.saved, 0)


class SummaryTests(unittest.TestCase):
    def test_summary_lists_blocked_hosts(self):
        rep = HostReputation()
        for _ in range(STRIKES_TO_BLOCK):
            rep.record("https://a.no/x", 403)
        for _ in range(STRIKES_TO_BLOCK):
            rep.record("https://b.no/x", 429)

        summary = rep.summary()
        self.assertEqual(summary["hosts_blocked"], 2)
        self.assertEqual(sorted(summary["blocked"]), ["a.no", "b.no"])

    def test_reset_clears_everything(self):
        rep = HostReputation()
        for _ in range(STRIKES_TO_BLOCK):
            rep.record("https://a.no/x", 403)
        rep.should_skip("https://a.no/y")
        rep.reset()

        self.assertFalse(rep.is_blocked("https://a.no"))
        self.assertEqual(rep.saved, 0)
        self.assertEqual(rep.summary()["hosts_blocked"], 0)


class IntegrationTests(unittest.TestCase):
    """The wiring: does a refusal actually stop the next request?"""

    def _responses(self, first_status=403, after=200):
        from unittest.mock import MagicMock

        def make(status):
            r = MagicMock()
            r.status_code = status
            r.headers = {}
            r.content = b"ok"
            r.encoding = "utf-8"
            r.close = MagicMock()
            r.raise_for_status = MagicMock()
            r.__enter__ = lambda s: s
            r.__exit__ = lambda *a: False
            r.iter_content = lambda chunk_size=65536: iter([b"ok"])
            return r

        return make, first_status, after

    def test_second_request_to_a_refusing_host_is_not_sent(self):
        from unittest.mock import patch

        from src.hostreputation import reputation
        from src.http import HostRefused, get_following_redirects

        reputation.reset()
        self.addCleanup(reputation.reset)

        from unittest.mock import MagicMock

        calls = []

        def fake_get(url, **kwargs):
            calls.append(url)
            r = MagicMock()
            r.status_code = 403
            r.headers = {}
            r.close = MagicMock()
            return r

        with patch("src.http.requests.get", fake_get), patch(
            "src.netguard.resolve_is_public", lambda host: True
        ):
            # Two refusals; blocking starts at the second.
            for _ in range(2):
                with contextlib.suppress(Exception):
                    get_following_redirects("https://gulesider.no/a")
            self.assertEqual(len(calls), 2)

            # Third call: must not reach the network at all.
            with self.assertRaises(HostRefused):
                get_following_redirects("https://gulesider.no/b")

        self.assertEqual(len(calls), 2, "a blocked host was requested again")

    def test_registry_traffic_is_not_reputation_tracked(self):
        # Brreg is never going to refuse us, and the registry has fixed
        # URLs, so nothing is gained by tracking it.
        from unittest.mock import patch

        from src.hostreputation import reputation
        from src.http import get_following_redirects

        reputation.reset()
        self.addCleanup(reputation.reset)

        from unittest.mock import MagicMock

        def fake_get(url, **kwargs):
            r = MagicMock()
            r.status_code = 403
            r.headers = {}
            r.close = MagicMock()
            return r

        with patch("src.http.requests.get", fake_get):
            for _ in range(4):
                with contextlib.suppress(Exception):
                    get_following_redirects(
                        "https://data.brreg.no/x", validate=False
                    )

        self.assertFalse(reputation.is_blocked("data.brreg.no"))


if __name__ == "__main__":
    unittest.main()
