"""
Redirect handling and SSRF.

The guard answers "may the agent open a connection to this host?" --
and the question has to be asked again at every hop, not only for the
URL the caller supplied.

Delegating redirects to the HTTP library meant it was asked once. A
discovered ``https://company.example`` that 302s to
``http://169.254.169.254/latest/meta-data/`` passed the check and
reached cloud instance metadata anyway. Validating the URL the agent
decided to check is worthless if the request it never made is the one
that goes out.

The agent's inputs are attacker-influenced: every page URL comes from a
search index, so this is reachable rather than theoretical.
"""

import unittest
from unittest.mock import MagicMock, patch

import requests

from src.http import get_following_redirects
from src.netguard import (
    DEFAULT_MAX_REDIRECTS,
    TooManyRedirects,
    UnsafeUrl,
    redirect_target,
    validate_url,
)


def _response(status_code=200, location=None, body=b"ok"):
    response = MagicMock()
    response.status_code = status_code
    response.headers = {"Location": location} if location else {}
    response.content = body
    response.encoding = "utf-8"
    response.close = MagicMock()
    response.__enter__ = lambda s: s
    response.__exit__ = lambda *a: False
    # fetch_bytes streams rather than reading .content.
    response.iter_content = lambda chunk_size=65536: iter([body])
    return response


def _resolver(*, everything_public: bool = False):
    """A stand-in for DNS resolution.

    ``everything_public=True`` is convenient for chain tests, but it must
    never be used when the point is that a private address is refused:
    it would make ``169.254.169.254`` resolve as public and quietly
    destroy the thing being tested.
    """
    if everything_public:
        return lambda host: True

    def resolve(host: str) -> bool:
        import ipaddress

        try:
            return not ipaddress.ip_address(host).is_private
        except ValueError:
            return True

    return resolve


class ValidateUrlTests(unittest.TestCase):
    def test_public_host_is_allowed(self):
        with patch("src.netguard.resolve_is_public", return_value=True):
            self.assertEqual(
                validate_url("https://example.no/about"),
                "https://example.no/about",
            )

    def test_private_targets_are_refused(self):
        for url in (
            "http://127.0.0.1/",
            "http://169.254.169.254/latest/meta-data/",
            "http://192.168.1.1/",
            "http://10.0.0.1/",
            "http://[::1]/",
        ):
            with self.assertRaises(UnsafeUrl, msg=url):
                validate_url(url)

    def test_non_http_schemes_are_refused(self):
        for url in ("file:///etc/passwd", "ftp://example.com/", "gopher://x"):
            with self.assertRaises(UnsafeUrl, msg=url):
                validate_url(url)

    def test_scheme_is_checked_before_coercion(self):
        """The order of operations, not just the outcome.

        Prepending "https://" first turned "file:///etc/passwd" into
        "https://file:///etc/passwd", which then parsed as scheme https
        and passed. The old test only caught it because the mangled host
        happened not to resolve.
        """
        with patch("src.netguard.resolve_is_public", return_value=True):
            # Even with every host resolving as public, a non-HTTP scheme
            # must still be refused on the scheme alone.
            for url in ("file:///etc/passwd", "ftp://example.com/x"):
                with self.assertRaises(UnsafeUrl, msg=url):
                    validate_url(url)

    def test_credentials_in_url_are_refused(self):
        # Would otherwise be written to the cache directory verbatim.
        with patch("src.netguard.resolve_is_public", return_value=True):
            with self.assertRaises(UnsafeUrl):
                validate_url("https://user:pass@example.no/x")

    def test_metadata_endpoint_is_refused(self):
        # The single highest-value target for an SSRF, called out
        # explicitly so it is never weakened by a refactor.
        for url in (
            "http://169.254.169.254/latest/meta-data/iam/",
            "http://metadata.google.internal/computeMetadata/v1/",
        ):
            with self.assertRaises(UnsafeUrl, msg=url):
                validate_url(url)


class RedirectTargetTests(unittest.TestCase):
    def test_non_redirect_returns_none(self):
        self.assertIsNone(redirect_target(_response(200), max_redirects=5))

    def test_redirect_returns_location(self):
        response = _response(302, location="https://example.no/x")
        self.assertEqual(
            redirect_target(response, max_redirects=5), "https://example.no/x"
        )

    def test_redirect_without_location_is_a_dead_end(self):
        self.assertIsNone(redirect_target(_response(302), max_redirects=5))

    def test_exhausted_budget_raises(self):
        response = _response(302, location="https://example.no/x")
        with self.assertRaises(TooManyRedirects):
            redirect_target(response, max_redirects=0)


class FollowTests(unittest.TestCase):
    """The behaviour the library's default hid."""

    def test_redirect_to_a_private_address_is_refused(self):
        """The bug. A safe URL redirecting somewhere unsafe."""
        hops = {
            "https://company.example": _response(
                302, location="http://169.254.169.254/latest/meta-data/"
            ),
        }

        def fake_get(url, **kwargs):
            # Critically, the library must not follow this itself.
            self.assertFalse(kwargs.get("allow_redirects", True))
            return hops[url]

        with patch("src.http.requests.get", fake_get), patch(
            "src.netguard.resolve_is_public", _resolver()
        ):
            with self.assertRaises(UnsafeUrl):
                get_following_redirects("https://company.example")

    def test_redirect_chain_is_followed_when_every_hop_is_safe(self):
        hops = {
            "https://a.example": _response(301, location="/next"),
            "https://a.example/next": _response(302, location="https://b.example/final"),
            "https://b.example/final": _response(200),
        }

        def fake_get(url, **kwargs):
            return hops[url]

        with patch("src.http.requests.get", fake_get), patch(
            "src.netguard.resolve_is_public", _resolver(everything_public=True)
        ):
            response = get_following_redirects(
                "https://a.example", max_redirects=5
            )

        self.assertEqual(response.status_code, 200)

    def test_loop_is_capped(self):
        def fake_get(url, **kwargs):
            return _response(302, location="https://loop.example/again")

        with patch("src.http.requests.get", fake_get), patch(
            "src.netguard.resolve_is_public", _resolver(everything_public=True)
        ):
            with self.assertRaises(TooManyRedirects):
                get_following_redirects("https://loop.example")

    def test_relative_redirect_resolves_against_the_current_url(self):
        hops = {
            "https://a.example/dir/page": _response(302, location="sibling"),
            "https://a.example/dir/sibling": _response(200),
        }

        def fake_get(url, **kwargs):
            return hops[url]

        with patch("src.http.requests.get", fake_get), patch(
            "src.netguard.resolve_is_public", _resolver(everything_public=True)
        ):
            response = get_following_redirects(
                "https://a.example/dir/page", max_redirects=5
            )

        self.assertEqual(response.status_code, 200)

    def test_redirect_to_a_non_http_scheme_is_refused(self):
        hops = {
            "https://a.example": _response(302, location="file:///etc/passwd"),
        }

        def fake_get(url, **kwargs):
            return hops[url]

        with patch("src.http.requests.get", fake_get), patch(
            "src.netguard.resolve_is_public", _resolver()
        ):
            with self.assertRaises(UnsafeUrl):
                get_following_redirects("https://a.example")

    def test_validation_can_be_skipped_for_the_registry(self):
        # Brreg endpoints are fixed strings from source, not from a search
        # index, so they skip the guard to avoid a DNS lookup per call.
        seen = {}

        def fake_get(url, **kwargs):
            seen["url"] = url
            return _response(200)

        with patch("src.http.requests.get", fake_get), patch(
            "src.netguard.resolve_is_public"
        ) as resolver:
            get_following_redirects(
                "https://data.brreg.no/x", validate=False
            )

        self.assertEqual(seen["url"], "https://data.brreg.no/x")
        resolver.assert_not_called()

    def test_default_cap_is_finite(self):
        self.assertGreater(DEFAULT_MAX_REDIRECTS, 0)
        self.assertLessEqual(DEFAULT_MAX_REDIRECTS, 10)


class FetchIntegrationTests(unittest.TestCase):
    def test_fetch_bytes_refuses_a_private_redirect(self):
        from src.http import fetch_bytes

        hops = {
            "https://company.example": _response(
                302, location="http://169.254.169.254/latest/meta-data/"
            ),
        }

        def fake_get(url, **kwargs):
            return hops[url]

        with patch("src.http.requests.get", fake_get), patch(
            "src.netguard.resolve_is_public", _resolver()
        ):
            with self.assertRaises(UnsafeUrl):
                fetch_bytes("https://company.example")

    def test_fetch_bytes_still_returns_a_body_when_safe(self):
        from src.http import ByteResponse, fetch_bytes

        def fake_get(url, **kwargs):
            return _response(200, body=b"<html>ok</html>")

        with patch("src.http.requests.get", fake_get), patch(
            "src.netguard.resolve_is_public", _resolver(everything_public=True)
        ):
            result = fetch_bytes("https://example.no/page")

        self.assertIsInstance(result, ByteResponse)
        self.assertEqual(result.body, b"<html>ok</html>")

    def test_transport_errors_are_retried_not_raised_as_unsafe(self):
        from src.http import fetch_json

        with patch(
            "src.http.get_following_redirects",
            side_effect=requests.RequestException("boom"),
        ):
            with self.assertRaises(requests.RequestException):
                fetch_json("https://data.brreg.no/x", attempts=2)


if __name__ == "__main__":
    unittest.main()
