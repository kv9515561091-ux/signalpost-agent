"""Tests for the reviewer-facing synthesis block and the run planner."""

from __future__ import annotations

import unittest

from src.contract import envelope_for


def profile(**overrides):
    base = {
        "company_number": "989061593",
        "company_name": "G Group AS",
        "legal_form": "AS",
        "retrieved_at": "2026-11-09T21:00:00Z",
        "explanation": {
            "text": "G Group AS is a Norwegian limited company.",
            "generated_by": "deterministic",
        },
        "evidence": [
            {
                "field": "company_name",
                "value": "G Group AS",
                "source_url": "https://data.brreg.no/x",
                "retrieved_at": "2026-11-09T21:00:00Z",
                "evidence_text": "navn: G Group AS",
                "confidence": 0.99,
                "source_tier": 1,
                "source_type": "official_registry",
            },
            {
                "field": "job_opening",
                "value": "Senior Engineer",
                "source_url": "https://arbeidsplassen.nav.no/stillinger/stilling/a",
                "retrieved_at": "2026-11-09T21:00:00Z",
                "evidence_text": "NAV Arbeidsplassen advertises ...",
                "confidence": 0.7,
                "source_tier": 4,
                "source_type": "official_job_feed",
                "as_of": "2026-11-09",
            },
        ],
        "availability": {"news": "not_available", "phone": "blocked"},
        "changes": [{"field": "revenue", "from": 1, "to": 2}],
    }
    base.update(overrides)
    return base


class SynthesisTests(unittest.TestCase):
    def build(self, data):
        envelope = envelope_for(
            data,
            run_id="run-1",
            started_at="2026-11-09T21:00:00Z",
            completed_at="2026-11-09T21:05:00Z",
        )
        return envelope["synthesis"]

    def test_claim_states_are_counted(self):
        synthesis = self.build(profile())
        self.assertEqual(synthesis["claim_count"], 4)
        self.assertEqual(synthesis["claim_states"]["available"], 2)
        self.assertEqual(synthesis["claim_states"]["not_available"], 1)
        self.assertEqual(synthesis["claim_states"]["blocked"], 1)

    def test_unknowns_name_the_gap_and_its_state(self):
        synthesis = self.build(profile())
        unknowns = {item["field"]: item["state"] for item in synthesis["unknowns"]}
        self.assertEqual(unknowns["news"], "not_available")
        self.assertEqual(unknowns["phone"], "blocked")

    def test_sources_are_grouped_by_contract_class(self):
        synthesis = self.build(profile())
        by_class = {
            item["source_class"]: item for item in synthesis["sources"]
        }
        self.assertEqual(by_class["government_registry"]["claims"], 1)
        self.assertEqual(by_class["third_party_database"]["claims"], 1)
        self.assertIn(
            "official_job_feed",
            by_class["third_party_database"]["source_types"],
        )

    def test_coverage_names_present_and_missing_families(self):
        coverage = self.build(profile())["coverage"]
        self.assertIn("identity", coverage["available_families"])
        self.assertIn("workforce", coverage["available_families"])
        self.assertIn("social", coverage["missing_families"])
        self.assertIn("financials", coverage["missing_families"])
        families = {item["family"]: item for item in coverage["families"]}
        self.assertEqual(families["identity"]["claims"], 1)
        self.assertEqual(families["social"]["claims"], 0)

    def test_highlights_lead_with_current_state(self):
        highlights = self.build(profile())["highlights"]
        self.assertTrue(highlights)
        self.assertTrue(highlights[0].startswith("job_opening:"))

    def test_changes_and_summary_are_kept(self):
        synthesis = self.build(profile())
        self.assertEqual(
            synthesis["changes_summary"],
            "1 field(s) changed since the previous run",
        )
        self.assertEqual(
            synthesis["summary"],
            "G Group AS is a Norwegian limited company.",
        )

    def test_empty_profile_still_has_the_shape(self):
        synthesis = self.build({"company_number": "989061593"})
        self.assertEqual(synthesis["claim_count"], 0)
        self.assertEqual(synthesis["sources"], [])
        self.assertEqual(synthesis["unknowns"], [])
        self.assertEqual(
            sorted(synthesis["coverage"]["missing_families"]),
            sorted(synthesis["coverage"]["missing_families"]),
        )
        self.assertIn("identity", synthesis["coverage"]["missing_families"])


class WebsiteCheckPlannerTests(unittest.TestCase):
    def plan(self, inputs, records, limit):
        from src.contract_cli import _select_website_checks

        return _select_website_checks(inputs, records, limit)

    def test_zero_disables_the_check(self):
        inputs = [{"organisation_number": "1"}]
        records = {"1": {"website": "https://a.no"}}
        self.assertEqual(self.plan(inputs, records, 0), set())

    def test_negative_means_every_company(self):
        self.assertIsNone(self.plan([{"organisation_number": "1"}], {}, -1))

    def test_companies_without_a_website_are_never_selected(self):
        inputs = [{"organisation_number": "1"}, {"organisation_number": "2"}]
        records = {
            "1": {"website": None, "employee_count": 900},
            "2": {"website": "https://b.no", "employee_count": 1},
        }
        self.assertEqual(self.plan(inputs, records, 5), {"2"})

    def test_largest_companies_come_first(self):
        inputs = [{"organisation_number": str(n)} for n in range(4)]
        records = {
            "0": {"website": "https://a.no", "employee_count": 5},
            "1": {"website": "https://b.no", "employee_count": 500},
            "2": {"website": "https://c.no", "employee_count": 50},
            "3": {"website": "https://d.no", "employee_count": 5000},
        }
        self.assertEqual(self.plan(inputs, records, 2), {"3", "1"})

    def test_limit_defaults_to_every_company_below_the_bulk_threshold(self):
        from src.contract_cli import (
            AUTO_NEWS_CHECKS,
            AUTO_WEBSITE_CHECKS,
            _news_check_limit,
            _website_check_limit,
        )

        self.assertEqual(_website_check_limit(100, None), 100)
        self.assertEqual(_website_check_limit(1000, None), AUTO_WEBSITE_CHECKS)
        self.assertEqual(_website_check_limit(1000, 0), 0)
        self.assertEqual(_website_check_limit(1000, -1), -1)

        self.assertEqual(_news_check_limit(100, None), 100)
        self.assertEqual(_news_check_limit(1000, None), AUTO_NEWS_CHECKS)
        self.assertEqual(_news_check_limit(1000, 0), 0)
        self.assertEqual(_news_check_limit(1000, -1), -1)

    def test_the_bulk_defaults_fit_inside_the_request_ceiling(self):
        from src.contract_cli import (
            AUTO_NEWS_CHECKS,
            AUTO_WEBSITE_CHECKS,
            AUTO_WIKIDATA_MIN,
        )

        # A 1,000-company run spends ~1,427 registry requests (roles and
        # filed accounts) and measured ~122 of retry overhead on failed
        # fetches, plus one batch Wikidata request. The defaults must
        # leave headroom so the tail is never dropped at the ceiling.
        wikidata = 1 if AUTO_WIKIDATA_MIN <= 1000 else 0
        self.assertLessEqual(
            1427 + 122 + AUTO_WEBSITE_CHECKS + AUTO_NEWS_CHECKS + wikidata,
            2000,
        )

    def test_a_larger_batch_shrinks_the_auto_enrichment_caps(self):
        from src.contract_cli import (
            AUTO_NEWS_CHECKS,
            AUTO_WEBSITE_CHECKS,
            RETRY_MARGIN,
            SAFETY_MARGIN,
            _fit_enrichment_to_ceiling,
        )

        inputs = [
            {"organisation_number": str(100000000 + i)} for i in range(1100)
        ]
        records = {
            str(100000000 + i): {"last_filed_annual_accounts": i % 2 == 0}
            for i in range(1100)
        }
        filings = 550
        registry_cost = len(inputs) + filings
        room = max(0, 2000 - registry_cost - RETRY_MARGIN - SAFETY_MARGIN)

        website, news = _fit_enrichment_to_ceiling(
            inputs, None, None, AUTO_WEBSITE_CHECKS, AUTO_NEWS_CHECKS, records, 2000
        )
        self.assertEqual(website, min(AUTO_WEBSITE_CHECKS, room * 74 // 100))
        self.assertEqual(news, min(AUTO_NEWS_CHECKS, max(0, room - website)))
        self.assertLessEqual(
            registry_cost + website + news + RETRY_MARGIN + SAFETY_MARGIN,
            2000,
        )

        # The verified 1,000-company defaults are untouched.
        self.assertEqual(
            _fit_enrichment_to_ceiling(
                inputs[:1000],
                None,
                None,
                AUTO_WEBSITE_CHECKS,
                AUTO_NEWS_CHECKS,
                records,
                2000,
            ),
            (AUTO_WEBSITE_CHECKS, AUTO_NEWS_CHECKS),
        )

        # Explicit overrides are honoured: a named number owns the budget.
        self.assertEqual(
            _fit_enrichment_to_ceiling(inputs, 5, 3, 5, 3, records, 2000),
            (5, 3),
        )


if __name__ == "__main__":
    unittest.main()
