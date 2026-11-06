"""Careers-page reading is deterministic, so its behaviour is testable."""

from __future__ import annotations

import unittest

from src.agent import _careers_probe
from src.jobs import careers_index_links, is_careers_url, read_careers_page


class CareersUrlTests(unittest.TestCase):
    def test_careers_paths_are_recognised(self):
        for url in (
            "https://example.no/careers",
            "https://example.no/karriere/",
            "https://example.no/jobs",
            "https://example.no/ledige-stillinger",
            "https://example.no/bli-med",
        ):
            with self.subTest(url=url):
                self.assertTrue(is_careers_url(url))

    def test_about_and_press_are_not_careers(self):
        for url in (
            "https://example.no/om-oss",
            "https://example.no/news",
            "https://example.no/careers-archive",
        ):
            with self.subTest(url=url):
                self.assertFalse(is_careers_url(url))


class CareersProbeTests(unittest.TestCase):
    def test_norwegian_domains_probe_karriere(self):
        self.assertEqual(
            _careers_probe("https://example.no/om-oss"),
            "https://example.no/karriere",
        )

    def test_other_domains_probe_careers(self):
        self.assertEqual(
            _careers_probe("example.com"),
            "https://example.com/careers",
        )

    def test_missing_website_has_no_probe(self):
        self.assertIsNone(_careers_probe(None))


class ReadCareersPageTests(unittest.TestCase):
    def test_posting_links_become_titles(self):
        findings = read_careers_page(
            "Ledige stillinger",
            "https://example.no/karriere",
            [
                "/stillinger/1234-senior-ingenior",
                "/stillinger/5678-prosjektleder",
                "/stillinger/1234-senior-ingenior",
                "/om-oss",
                "mailto:jobb@example.no",
            ],
        )
        self.assertEqual(findings.posting_links, 2)
        self.assertEqual(
            findings.titles,
            ["Senior Ingenior", "Prosjektleder"],
        )

    def test_ats_links_count_as_postings(self):
        findings = read_careers_page(
            "",
            "https://example.no/careers",
            ["https://example.teamtailor.com/jobs/999"],
        )
        self.assertEqual(findings.posting_links, 1)

    def test_checked_page_without_openings_is_empty_not_an_error(self):
        findings = read_careers_page(
            "Vi har ingen ledige stillinger akkurat n\u00e5.",
            "https://example.no/karriere",
            ["/kontakt"],
        )
        self.assertTrue(findings.checked)
        self.assertEqual(findings.titles, [])
        self.assertEqual(findings.posting_links, 0)
        self.assertIn("ingen ledige stillinger", findings.snippet)


class CareersIndexTests(unittest.TestCase):
    """A vacancy index is a link to more pages, not an opening."""

    def test_index_link_is_not_counted_as_a_posting(self):
        findings = read_careers_page(
            "",
            "https://example.no/careers",
            ["/careers/vacancies/", "/careers/vacancies/1-senior-utvikler/"],
        )
        self.assertEqual(findings.posting_links, 1)
        self.assertEqual(findings.titles, ["Senior Utvikler"])

    def test_ats_sign_in_links_do_not_become_titles(self):
        findings = read_careers_page(
            "",
            "https://example.no/careers",
            [
                "https://acme.hr-manager.net/Agent/Subscription.aspx?customer=acme",
                "https://acme.easycruit.com/career-center/login?iso=gb",
            ],
        )
        self.assertEqual(findings.posting_links, 2)
        self.assertEqual(findings.titles, [])

    def test_index_links_are_returned_in_page_order(self):
        self.assertEqual(
            careers_index_links(
                "https://example.no/karriere",
                [
                    "/karriere/ledige-stillinger",
                    "/karriere/ledige-stillinger/1234-senior-ingenior",
                    "/om-oss",
                    "https://jobs.example.com/jobs/",
                ],
            ),
            [
                "https://example.no/karriere/ledige-stillinger",
                "https://jobs.example.com/jobs/",
            ],
        )

    def test_page_cannot_index_itself(self):
        self.assertEqual(
            careers_index_links(
                "https://example.no/karriere",
                ["/karriere/", "/karriere"],
            ),
            [],
        )

    def test_index_links_are_capped(self):
        links = careers_index_links(
            "https://example.no/karriere",
            [
                "/karriere/ledige-stillinger",
                "/karriere/jobb",
                "/karriere/stillinger",
            ],
            max_links=2,
        )
        self.assertEqual(len(links), 2)

    def test_listing_page_does_not_chain_further_listings(self):
        self.assertEqual(
            careers_index_links(
                "https://www.example.no/careers/vacancies/",
                [
                    "/careers/vacancies/all",
                    "/careers/ledige-stillinger",
                ],
                has_postings=True,
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
