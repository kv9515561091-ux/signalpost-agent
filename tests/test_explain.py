"""Explanation: the deterministic path must never depend on an LLM."""

import unittest
from unittest.mock import patch

from src import llm
from src.explain import Explanation, build_deterministic, explain
from src.agent import CompanyProfile, Evidence, FactRecord, FinancialYear


def profile(**overrides):
    base = dict(
        company_number="989061593",
        company_name="-G- GROUP AS",
        legal_form="Aksjeselskap",
        status="aktiv",
        industry="IT-konsulentvirksomhet",
        industry_code="62.200",
        registration_date="2006-01-23",
        founding_date="2005-10-01",
        address="c/o Gjørva, Hellerud gårdsvei 5B, 0671 OSLO, Norge",
        business_description="Investeringer og eiendom.",
        employee_count=0,
        employee_count_as_of="2026-09-14",
        ceo_name="Geir Inge Gjørva",
        owner_name=None,
        board_chair_name="Geir Inge Gjørva",
        board_size=2,
        auditor_name=None,
        bankruptcy=False,
        under_liquidation=False,
        financial_currency="NOK",
        financials=[
            FinancialYear(
                fiscal_year="2025",
                currency="NOK",
                statement_scope="SELSKAP",
                revenue=20000.0,
                net_result=264344.0,
                total_assets=2949837.0,
            )
        ],
        sources=["https://data.brreg.no/x"],
        evidence=[
            Evidence(
                field="company_name",
                value="-G- GROUP AS",
                source_url="https://data.brreg.no/x",
                retrieved_at="2026-10-13T00:00:00+00:00",
                evidence_text="registered",
                confidence=1.0,
            )
        ],
        facts=[],
        conflicts=[],
        retrieval_errors=[],
        extraction_rejected=[],
        explanation=None,
    )
    base.update(overrides)
    return CompanyProfile(**base)


class DeterministicTests(unittest.TestCase):
    def test_states_identity_and_provenance(self):
        text = build_deterministic(profile())
        self.assertIn("989061593", text)
        self.assertIn("Aksjeselskap", text)
        self.assertIn("Hellerud gårdsvei", text)
        self.assertIn("Brønnøysundregistrene", text)

    def test_reports_management(self):
        text = build_deterministic(profile())
        self.assertIn("Geir Inge Gjørva", text)

    def test_sole_proprietor_is_not_reported_as_missing_ceo(self):
        text = build_deterministic(
            profile(ceo_name=None, owner_name="Andrii Zotko")
        )
        self.assertIn("Andrii Zotko", text)
        self.assertIn("sole owner", text)

    def test_reports_absent_financials_explicitly(self):
        text = build_deterministic(profile(financials=[], financial_currency=None))
        self.assertIn("No annual accounts", text)

    def test_surfaces_retrieval_gaps_as_not_retrieved(self):
        text = build_deterministic(profile(retrieval_errors=["roles: ReadTimeout"]))
        self.assertIn("ReadTimeout", text)
        # A gap must be distinguishable from the company having nothing.
        self.assertIn("should not be read as absence of the fact", text)

    def test_surfaces_conflicts(self):
        text = build_deterministic(
            profile(conflicts=[{"field": "employee_count"}])
        )
        self.assertIn("differently by different sources", text)

    def test_flags_bankruptcy(self):
        text = build_deterministic(profile(bankruptcy=True, status=None))
        self.assertIn("bankrupt", text)

    def test_mentions_extracted_facts(self):
        fact = FactRecord(
            field="services",
            value="Advisory",
            source_url="https://example.no",
            retrieved_at="2026-10-13T00:00:00+00:00",
            evidence_text="quote",
            confidence=0.9,
            source_tier=2,
        )
        text = build_deterministic(profile(facts=[fact]))
        self.assertIn("1 additional fact", text)


class FallbackTests(unittest.TestCase):
    def test_no_llm_yields_deterministic_text(self):
        with patch("src.explain.llm.is_available", return_value=False):
            with patch(
                "src.explain.llm.unavailable_reason", return_value="no key"
            ):
                result = explain(profile(), use_llm=True)
        self.assertFalse(result.llm_used)
        self.assertEqual(result.generated_by, "deterministic")
        self.assertIn("no key", result.note)
        self.assertIn("989061593", result.text)

    def test_use_llm_false_skips_entirely(self):
        result = explain(profile(), use_llm=False)
        self.assertFalse(result.llm_used)
        self.assertEqual(result.generated_by, "deterministic")

    def test_llm_failure_falls_back(self):
        with patch("src.explain.llm.is_available", return_value=True):
            with patch(
                "src.explain.llm.complete_text",
                side_effect=llm.LLMCallFailed("timeout"),
            ):
                result = explain(profile(), use_llm=True)
        self.assertFalse(result.llm_used)
        self.assertIn("unavailable", result.note)

    def test_successful_llm_narrative_is_prepended(self):
        with patch("src.explain.llm.is_available", return_value=True):
            with patch(
                "src.explain.llm.complete_text",
                return_value="The company invests in property in Oslo.",
            ):
                result = explain(profile(), use_llm=True)
        self.assertTrue(result.llm_used)
        # Attribution names whichever provider answered.
        self.assertIn("openai_compatible", result.generated_by)
        self.assertTrue(result.text.startswith("The company invests"))
        # The provenance section is still appended from the record.
        self.assertIn("Brønnøysundregistrene", result.text)

    def test_narrative_citing_another_organization_is_rejected(self):
        with patch("src.explain.llm.is_available", return_value=True):
            with patch(
                "src.explain.llm.complete_text",
                return_value="This is actually 983887457 doing the work.",
            ):
                result = explain(profile(), use_llm=True)
        self.assertFalse(result.llm_used)
        self.assertIn("another organization", result.note)
        self.assertNotIn("This is actually", result.text)


class SerialisationTests(unittest.TestCase):
    def test_explanation_serialises(self):
        result = Explanation(text="t", generated_by="deterministic")
        self.assertEqual(result.to_dict()["generated_by"], "deterministic")
        self.assertIn("text", result.to_dict())


if __name__ == "__main__":
    unittest.main()