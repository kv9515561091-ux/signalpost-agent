"""Tests for the offline static-site renderer."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from src.contract import envelope_for
from src.html_report import render_site


def _envelope(
    org: str,
    name: str,
    *,
    web_malicious: bool = False,
    state: str | None = None,
) -> dict[str, Any]:
    if state:
        profile: dict[str, Any] = {
            "company_number": org,
            "company_name": name,
            "error": "ConnectionError: ('Connection aborted.', ConnectionResetError(10054))",
        }
    else:
        profile = {
            "company_number": org,
            "organisation_number": org,
            "company_name": name,
            "legal_form": "Aksjeselskap",
            "industry": "Databehandling",
            "retrieved_at": "2026-11-12T00:00:00Z",
            "website": "javascript:alert(1)" if web_malicious else "https://example.no",
            "website_verified": True,
            "evidence": [
                {
                    "field": "company_name",
                    "value": name,
                    "source_url": "https://data.brreg.no/enhetsregisteret/api/enheter/" + org,
                    "retrieved_at": "2026-11-12T00:00:00Z",
                    "evidence_text": f"org.nr {org} {name}",
                    "confidence": 0.99,
                    "source_tier": 1,
                    "source_type": "official_registry",
                },
                {
                    "field": "revenue",
                    "value": 12_000_000,
                    "source_url": "https://data.brreg.no/regnskapsregisteret/regnskap/2025/" + org,
                    "retrieved_at": "2026-11-12T00:00:00Z",
                    "evidence_text": f"regnskap {org}",
                    "confidence": 0.85,
                    "source_tier": 3,
                    "source_type": "official_annual_accounts",
                    "as_of": "2025-12-31",
                },
            ],
            "availability": {"news": "not_available"},
            "changes": [],
        }
    return envelope_for(
        profile,
        run_id="run-1",
        started_at="2026-11-12T00:00:00Z",
        completed_at="2026-11-12T00:05:00Z",
    )


class HtmlReportTests(unittest.TestCase):
    def test_every_company_gets_a_page_and_index_links_them(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            envelopes = [
                _envelope("100000001", "Alpha AS"),
                _envelope("100000002", "Beta & Sønn <AS>"),
                _envelope(
                    "100000003",
                    "Gamma",
                    state="error",
                ),
            ]
            result = render_site(envelopes, out)
            self.assertEqual(result["pages"], 3)
            self.assertTrue((out / "index.html").exists())
            index = (out / "index.html").read_text(encoding="utf-8")
            for org in ("100000001", "100000002", "100000003"):
                self.assertTrue((out / f"{org}.html").exists())
            self.assertIn("Alpha AS", index)
            self.assertIn("100000001", index)

    def test_registry_text_is_escaped_everywhere(self):
        name = "Evil <script>alert('x')</script> & Co"
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            render_site([_envelope("100000004", name)], out)
            index = (out / "index.html").read_text(encoding="utf-8")
            self.assertNotIn("<script>alert('x')</script>", index)
            self.assertIn("&lt;script&gt;", index)
            for page in out.glob("*.html"):
                text = page.read_text(encoding="utf-8")
                self.assertNotIn("<script>alert('x')</script>", text)

    def test_source_urls_must_be_http_or_dropped(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            render_site([_envelope("100000005", "Web AS", web_malicious=True)], out)
            page = (out / "100000005.html").read_text(encoding="utf-8")
            self.assertNotIn("javascript:", page)

    def test_abstentions_render_as_not_found_not_zero(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            render_site([_envelope("100000006", "Absent News AS")], out)
            page = (out / "100000006.html").read_text(encoding="utf-8")
            self.assertIn("not_available", page)
            self.assertNotIn("0 openings", page)

    def test_render_is_deterministic(self):
        envelopes = [_envelope("100000007", "Deterministic AS")]
        with TemporaryDirectory() as tmp_a, TemporaryDirectory() as tmp_b:
            render_site(envelopes, tmp_a, generated="2026-11-12T00:00:00Z")
            render_site(envelopes, tmp_b, generated="2026-11-12T00:00:00Z")
            for name in ("index.html", "100000007.html"):
                self.assertEqual(
                    Path(tmp_a, name).read_bytes(),
                    Path(tmp_b, name).read_bytes(),
                )

    def test_empty_input_still_writes_an_index(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            result = render_site([], out)
            self.assertEqual(result["pages"], 0)
            self.assertTrue((out / "index.html").exists())


if __name__ == "__main__":
    unittest.main()
