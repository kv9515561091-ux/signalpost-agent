"""Page fetching: HTML flattening and the SSRF guard."""

import unittest
from unittest.mock import patch

from src.http import ByteResponse
from src.page import UnsafeUrl, html_to_text, validate_url

HTML = """<!DOCTYPE html>
<html><head>
<title>  -G- GROUP AS  </title>
<style>body{color:red}</style>
<script>var tracking = 1;</script>
</head>
<body>
<h1>Om oss</h1>
<p>Vi er et <b>aksjeselskap</b> i Oslo.</p>
<p>Kontakt: post@example.no</p>
<noscript>enable js</noscript>
</body></html>"""


class HtmlToTextTests(unittest.TestCase):
    def test_extracts_title(self):
        title, _ = html_to_text(HTML)
        self.assertEqual(title, "-G- GROUP AS")

    def test_removes_script_and_style_bodies(self):
        _, text = html_to_text(HTML)
        self.assertNotIn("color:red", text)
        self.assertNotIn("var tracking", text)
        self.assertNotIn("enable js", text)

    def test_keeps_visible_text(self):
        _, text = html_to_text(HTML)
        self.assertIn("Vi er et aksjeselskap i Oslo.", text)
        self.assertIn("post@example.no", text)

    def test_decodes_entities(self):
        _, text = html_to_text("<p>Gj&oslash;rva &amp; S&oslash;lvberg</p>")
        self.assertIn("Gjørva & Sølvberg", text)

    def test_preserves_norwegian_characters(self):
        _, text = html_to_text("<p>Hellerud gårdsvei 5B, 0671 OSLO</p>")
        self.assertIn("gårdsvei", text)

    def test_handles_empty_input(self):
        title, text = html_to_text("")
        self.assertEqual(title, "")
        self.assertEqual(text, "")


class UrlGuardTests(unittest.TestCase):
    def test_blocks_private_and_local_addresses(self):
        for url in (
            "http://127.0.0.1/",
            "http://localhost/admin",
            "http://169.254.169.254/latest/meta-data/",
            "http://192.168.0.1/",
            "http://10.0.0.5/",
            "http://[::1]/",
        ):
            with self.assertRaises(UnsafeUrl, msg=url):
                validate_url(url)

    def test_blocks_non_http_schemes(self):
        for url in ("file:///etc/passwd", "ftp://example.com/x",
                    "gopher://example.com"):
            with self.assertRaises(UnsafeUrl, msg=url):
                validate_url(url)

    def test_blocks_empty_and_non_string(self):
        for bad in ("", None, 123):
            with self.assertRaises(UnsafeUrl):
                validate_url(bad)

    def test_adds_scheme_when_missing(self):
        with patch("src.netguard.resolve_is_public", return_value=True):
            self.assertEqual(
                validate_url("example.no/about"), "https://example.no/about"
            )

    def test_allows_public_host(self):
        self.assertTrue(
            validate_url("https://www.regjeringen.no/no/dep/aid/kontakt/id173/")
        )

    def test_unresolvable_host_is_refused(self):
        with self.assertRaises(UnsafeUrl):
            validate_url("https://this-host-does-not-exist-9x8y7z.no/")


class FetchPageTests(unittest.TestCase):
    @patch("src.page.fetch_bytes")
    def test_flattens_response_body(self, mock_fetch):
        mock_fetch.return_value = ByteResponse(
            body=HTML.encode("utf-8"),
            encoding="utf-8",
            content_type="text/html",
            truncated=False,
            status_code=200,
        )
        from src.page import fetch_page

        with patch("src.netguard.resolve_is_public", return_value=True):
            page = fetch_page("https://example.no/about")

        self.assertIn("Om oss", page.text)
        self.assertEqual(page.title.strip(), "-G- GROUP AS")
        self.assertFalse(page.truncated)

    @patch("src.page.fetch_bytes")
    def test_flags_truncation(self, mock_fetch):
        mock_fetch.return_value = ByteResponse(
            body=HTML.encode("utf-8"),
            encoding="utf-8",
            content_type="text/html",
            truncated=True,
            status_code=200,
        )
        from src.page import fetch_page

        with patch("src.netguard.resolve_is_public", return_value=True):
            page = fetch_page("https://example.no/about")

        self.assertTrue(page.truncated)


if __name__ == "__main__":
    unittest.main()
