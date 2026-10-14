"""
Extraction verification — the anti-hallucination guard.

These tests are the most important in the project. A model that invents a
fact is the failure this whole agent exists to prevent, and the only
defence is that its claims are checked against the page.
"""

import unittest
from unittest.mock import patch

from src import llm
from src.extract import (
    EXTRACTABLE_FIELDS,
    EXTRACTION_SCHEMA,
    RESERVED_FIELDS,
    extract_from_page,
    quote_exists,
    verify_candidate,
)

PAGE = """
- G- GROUP AS
About us
We are a Norwegian investment company founded in 2005.
We invest in real estate and provide financial advisory services
to businesses across Norway.
Our head office is in Oslo.
Org.nr: 989061593
Contact: post@example.no
"""


def candidate(field="business_summary", value="Investment company",
              quote="We are a Norwegian investment company founded in 2005.",
              published_date=""):
    return {
        "field": field,
        "value": value,
        "quote": quote,
        "published_date": published_date,
    }


def run_verify(cand, page_text=PAGE, registry_values=(), accepted=()):
    return verify_candidate(
        cand,
        page_text=page_text,
        source_url="https://example.no/about",
        retrieved_at="2026-10-13T00:00:00+00:00",
        tier=2,
        source_type="official_company_website",
        company_number="989061593",
        registry_values=registry_values,
        accepted=accepted,
    )


class QuoteVerificationTests(unittest.TestCase):
    def test_exact_quote_is_accepted(self):
        fact, reason = run_verify(candidate())
        self.assertIsNone(reason)
        self.assertIsNotNone(fact)
        self.assertEqual(fact.field, "business_summary")

    def test_paraphrased_quote_is_rejected(self):
        _, reason = run_verify(
            candidate(quote="The company invests in property and advisory.")
        )
        self.assertEqual(reason, "quote not found in page")

    def test_invented_quote_is_rejected(self):
        _, reason = run_verify(
            candidate(quote="We operate 40 offices across Scandinavia.")
        )
        self.assertEqual(reason, "quote not found in page")

    def test_empty_quote_is_rejected(self):
        _, reason = run_verify(candidate(quote=""))
        self.assertEqual(reason, "quote not found in page")

    def test_whitespace_differences_are_tolerated(self):
        self.assertTrue(
            quote_exists(
                "Norwegian investment company founded in 2005",
                "We are a   Norwegian\ninvestment\tcompany founded in 2005.",
            )
        )

    def test_diacritic_differences_are_tolerated(self):
        self.assertTrue(
            quote_exists("Hellerud Gardsvei", "Hellerud Gårdsvei")
        )

    def test_quote_citing_another_organization_is_rejected(self):
        # The page really does contain a second org number. A quote that
        # lifts it describes another entity, whatever else it says.
        page = (
            PAGE
            + "\nUnrelated reference: Org.nr: 983887457 is mentioned in our archive.\n"
        )
        fact, reason = run_verify(
            candidate(quote="Org.nr: 983887457 is mentioned in our archive."),
            page_text=page,
        )
        self.assertIsNone(fact)
        self.assertIn("another organization", reason)
        self.assertIn("983887457", reason)

    def test_foreign_number_elsewhere_on_page_is_not_fatal(self):
        # Only the quote is checked. A number elsewhere on the page does
        # not disqualify an otherwise sound fact.
        page = (
            PAGE
            + "\nUnrelated reference: Org.nr: 983887457 is mentioned in our archive.\n"
        )
        fact, reason = run_verify(
            candidate(quote="We are a Norwegian investment company founded in 2005."),
            page_text=page,
        )
        self.assertIsNone(reason)
        self.assertIsNotNone(fact)

    def test_target_number_in_quote_is_allowed(self):
        fact, reason = run_verify(candidate(quote="Org.nr: 989061593"))
        self.assertIsNone(reason)
        self.assertIsNotNone(fact)


class FieldAllowlistTests(unittest.TestCase):
    def test_prompt_excludes_personal_biography(self):
        from src.extract import SYSTEM_PROMPT

        lowered = SYSTEM_PROMPT.lower()
        self.assertIn("business", lowered)
        self.assertIn("biograph", lowered)

    def test_address_fields_are_refused_as_registry_owned(self):
        # A live run spent a model call rediscovering the registered
        # address. Restating registry data wastes the extraction slot.
        for field in ("locations", "geographic_presence", "address"):
            _, reason = run_verify(
                candidate(
                    field=field,
                    quote="We are a Norwegian investment company founded in 2005.",
                )
            )
            self.assertIn("registry", reason, field)

    def test_address_fields_are_not_even_offered(self):
        self.assertNotIn("locations", EXTRACTABLE_FIELDS)
        self.assertNotIn("geographic_presence", EXTRACTABLE_FIELDS)

    def test_extractable_fields_are_all_non_registry(self):
        # Every offered field must be something Brreg cannot supply,
        # otherwise the model is being invited to repeat it.
        self.assertEqual(set(EXTRACTABLE_FIELDS) & RESERVED_FIELDS, set())

    def test_extraction_vocabulary_targets_new_information(self):
        for field in (
            "business_summary",
            "products",
            "services",
            "target_customers",
            "recent_developments",
        ):
            self.assertIn(field, EXTRACTABLE_FIELDS)

    def test_prompt_warns_against_repeating_registry_data(self):
        from src.extract import SYSTEM_PROMPT

        self.assertIn("registry", SYSTEM_PROMPT.lower())
        self.assertIn("address", SYSTEM_PROMPT.lower())

    def test_registry_owned_fields_are_refused(self):
        # The model must never be able to restate identity.
        for reserved in sorted(RESERVED_FIELDS)[:8]:
            _, reason = run_verify(candidate(field=reserved))
            self.assertIsNotNone(reason, reserved)

    def test_unknown_field_is_refused(self):
        _, reason = run_verify(candidate(field="invented_field"))
        self.assertIn("not extractable", reason)

    def test_empty_value_is_refused(self):
        _, reason = run_verify(candidate(value="   "))
        self.assertEqual(reason, "empty value")


class RedundancyTests(unittest.TestCase):
    """Directory sites reprint registry fields; that is not a discovery."""

    def test_fact_restating_the_registered_address_is_refused(self):
        _, reason = run_verify(
            candidate(
                field="business_summary",
                value="The company is registered at Bekkeliveien 5, 3470 Slemmestad",
                quote="We are a Norwegian investment company founded in 2005.",
            ),
            registry_values=("Bekkeliveien 5, 3470 Slemmestad",),
        )
        self.assertIn("repeats a value already sourced", reason)

    def test_genuinely_new_fact_survives_alongside_registry_values(self):
        fact, reason = run_verify(
            candidate(
                field="business_summary",
                value="Founded in summer 2004 by a former banking programmer",
                quote="We are a Norwegian investment company founded in 2005.",
            ),
            registry_values=("Bekkeliveien 5, 3470 Slemmestad",),
        )
        self.assertIsNone(reason)
        self.assertIsNotNone(fact)

    def test_near_duplicate_facts_are_refused(self):
        # A live run produced both "attracting visitors from 148
        # countries" and "Visitors have come from 148 countries".
        first = candidate(
            field="services",
            value="The website hosts railway pictures attracting visitors from 148 countries",
            quote="We are a Norwegian investment company founded in 2005.",
        )
        fact, reason = run_verify(first)
        self.assertIsNone(reason)

        second = candidate(
            field="services",
            value="Visitors have come from 148 countries to the website",
            quote="We are a Norwegian investment company founded in 2005.",
        )
        fact2, reason2 = run_verify(second, accepted=[fact])
        self.assertIsNone(fact2)
        self.assertIn("duplicates", reason2)

    def test_different_facts_are_both_kept(self):
        a = candidate(
            field="services",
            value="Runs a large collection of Norwegian railway photographs",
            quote="We are a Norwegian investment company founded in 2005.",
        )
        fact, _ = run_verify(a)
        b = candidate(
            field="services",
            value="Serves corporate clients across the Nordic region",
            quote="We are a Norwegian investment company founded in 2005.",
        )
        fact2, reason2 = run_verify(b, accepted=[fact])
        self.assertIsNone(reason2)
        self.assertIsNotNone(fact2)

    def test_already_accepted_facts_block_cross_page_duplicates(self):
        # Two directories stating the same thing must publish it once.
        prior = run_verify(
            candidate(
                field="services",
                value="IT consulting, database development, courses and "
                      "translation services",
                quote="We are a Norwegian investment company founded in 2005.",
            )
        )[0]
        self.assertIsNotNone(prior)

        fact, reason = run_verify(
            candidate(
                field="services",
                value="IT consulting, database development, courses and "
                      "translation services offered nationwide",
                quote="We are a Norwegian investment company founded in 2005.",
            ),
            accepted=[prior],
        )
        self.assertIsNone(fact)
        self.assertIn("duplicates", reason)

    def test_short_values_do_not_trigger_false_matches(self):
        # A two-letter registry token must not match everything.
        fact, reason = run_verify(
            candidate(field="business_summary", value="Operates nationally"),
            registry_values=("AS",),
        )
        self.assertIsNone(reason)


class ValidateAndSchemaTests(unittest.TestCase):
    def test_every_object_sets_additional_properties_false(self):
        # The API requires this on every object in a structured schema.
        def walk(node):
            if isinstance(node, dict):
                if node.get("type") == "object":
                    self.assertIs(node.get("additionalProperties"), False)
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(EXTRACTION_SCHEMA)

    def test_schema_only_offers_extractable_fields(self):
        offered = EXTRACTION_SCHEMA["properties"]["facts"]["items"][
            "properties"
        ]["field"]["enum"]
        self.assertEqual(sorted(offered), sorted(EXTRACTABLE_FIELDS))

    def test_no_reserved_field_is_offered(self):
        self.assertFalse(set(EXTRACTABLE_FIELDS) & RESERVED_FIELDS)


class ExtractPipelineTests(unittest.TestCase):
    def test_unavailable_llm_yields_no_facts_and_says_why(self):
        with patch("src.extract.llm.is_available", return_value=False):
            with patch(
                "src.extract.llm.unavailable_reason",
                return_value="no key configured",
            ):
                report = extract_from_page(
                    page_text=PAGE,
                    source_url="https://example.no/about",
                    retrieved_at="2026-10-13T00:00:00+00:00",
                    company_number="989061593",
                    company_name="-G- GROUP AS",
                    tier=2,
                    source_type="official_company_website",
                )
        self.assertEqual(report.facts, [])
        self.assertFalse(report.llm_used)
        self.assertEqual(report.error, "no key configured")

    def test_empty_page_does_not_call_the_model(self):
        report = extract_from_page(
            page_text="   ",
            source_url="https://example.no",
            retrieved_at="2026-10-13T00:00:00+00:00",
            company_number="989061593",
            company_name="-G- GROUP AS",
            tier=2,
            source_type="official_company_website",
        )
        self.assertIn("no text content", report.error)

    def _run_with_model(self, payload):
        with patch("src.extract.llm.is_available", return_value=True):
            with patch(
                "src.extract.llm.complete_json",
                return_value=llm.LLMResult(
                    data=payload,
                    model="m",
                    provider="anthropic",
                    finish_reason="end_turn",
                ),
            ):
                return extract_from_page(
                    page_text=PAGE,
                    source_url="https://example.no/about",
                    retrieved_at="2026-10-13T00:00:00+00:00",
                    company_number="989061593",
                    company_name="-G- GROUP AS",
                    tier=2,
                    source_type="official_company_website",
                )

    def test_good_and_bad_facts_are_separated(self):
        report = self._run_with_model(
            {
                "facts": [
                    candidate(field="business_summary", value="Investment company"),
                    candidate(
                        field="services",
                        value="Advisory",
                        quote="We run 12 subsidiaries in Sweden.",
                    ),
                ]
            }
        )
        self.assertTrue(report.llm_used)
        self.assertEqual(len(report.facts), 1)
        self.assertEqual(report.facts[0].field, "business_summary")
        self.assertEqual(len(report.rejected), 1)
        self.assertEqual(report.rejected[0]["reason"], "quote not found in page")

    def test_confidence_is_capped_by_source_tier(self):
        report = self._run_with_model(
            {"facts": [candidate(field="business_summary")]}
        )
        self.assertEqual(report.facts[0].confidence, 0.90)

    def test_published_date_is_normalised(self):
        report = self._run_with_model(
            {"facts": [candidate(field="business_summary",
                                 published_date="Oppdatert 12. mars 2025")]},
        )
        self.assertEqual(report.facts[0].published_date, "2025-03-12")

    def test_norwegian_numeric_date_is_parsed(self):
        report = self._run_with_model(
            {"facts": [candidate(field="business_summary",
                                 published_date="Sist oppdatert 12.03.2025")]},
        )
        self.assertEqual(report.facts[0].published_date, "2025-03-12")

    def test_unparseable_date_is_dropped_not_guessed(self):
        report = self._run_with_model(
            {"facts": [candidate(field="business_summary",
                                 published_date="last Tuesday")]},
        )
        self.assertIsNone(report.facts[0].published_date)

    def test_refusal_is_reported_not_raised(self):
        with patch("src.extract.llm.is_available", return_value=True):
            with patch(
                "src.extract.llm.complete_json",
                side_effect=llm.LLMRefused("model declined"),
            ):
                report = extract_from_page(
                    page_text=PAGE,
                    source_url="https://example.no/about",
                    retrieved_at="2026-10-13T00:00:00+00:00",
                    company_number="989061593",
                    company_name="-G- GROUP AS",
                    tier=2,
                    source_type="official_company_website",
                )
        self.assertIn("refused", report.error)
        self.assertEqual(report.facts, [])


if __name__ == "__main__":
    unittest.main()