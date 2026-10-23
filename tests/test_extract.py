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
    BATCH_SYSTEM_PROMPT,
    EXTRACTABLE_FIELDS,
    EXTRACTION_SCHEMA,
    MAX_BATCH_PAGE_CHARS,
    RESERVED_FIELDS,
    PageInput,
    extract_from_page,
    extract_from_pages,
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


NEWS_PAGE = """
- G- GROUP AS
Nyheter
12. mars 2025
G-Group kjopte Example AS.
Sist oppdatert 13. mars 2025
"""


def news_candidate(
    value="G-Group kjopte Example AS.",
    quote="12. mars 2025\nG-Group kjopte Example AS.",
    published_date="2025-03-12",
):
    return candidate(
        field="recent_developments",
        value=value,
        quote=quote,
        published_date=published_date,
    )


def verify_news(cand, *, page_text=NEWS_PAGE,
                source_url="https://example.no/nyheter"):
    return verify_candidate(
        cand,
        page_text=page_text,
        source_url=source_url,
        retrieved_at="2026-10-13T00:00:00+00:00",
        tier=2,
        source_type="official_company_website",
        company_number="989061593",
        registry_values=(),
        accepted=(),
    )


class DatedNewsTests(unittest.TestCase):
    """A news item needs a real publication date in its own quote."""

    def test_dated_news_is_accepted(self):
        fact, reason = verify_news(news_candidate())
        self.assertIsNone(reason)
        self.assertEqual(fact.published_date, "2025-03-12")

    def test_missing_publication_date_is_rejected(self):
        _, reason = verify_news(
            news_candidate(published_date="")
        )
        self.assertIn("require a stated publication date", reason)

    def test_date_outside_the_quote_is_rejected(self):
        _, reason = verify_news(
            news_candidate(
                quote="G-Group kjopte Example AS.",
                published_date="2025-03-12",
            )
        )
        self.assertIn("not present in the quoted text", reason)

    def test_modified_timestamp_is_not_a_publication_date(self):
        _, reason = verify_news(
            news_candidate(
                quote="Sist oppdatert 13. mars 2025",
                published_date="2025-03-13",
            )
        )
        self.assertIn("modified or updated timestamp", reason)

    def test_feed_updated_timestamp_is_rejected(self):
        page = "Updated 12 March 2025\nG-Group kjopte Example AS."
        _, reason = verify_news(
            news_candidate(
                quote="Updated 12 March 2025\nG-Group kjopte Example AS.",
            ),
            page_text=page,
        )
        self.assertIn("modified or updated timestamp", reason)

    def test_article_using_updated_is_not_a_timestamp(self):
        # "Updated" in ordinary prose must not disqualify a dated story;
        # only an update/date pair marks a modified timestamp.
        page = (
            "G-Group updated its guidance on 12 March 2025 after the "
            "contract.\n12 March 2025"
        )
        _, reason = verify_news(
            news_candidate(
                value="G-Group updated its guidance after the contract.",
                quote=(
                    "G-Group updated its guidance on 12 March 2025 after "
                    "the contract."
                ),
                published_date="2025-03-12",
            ),
            page_text=page,
        )
        self.assertIsNone(reason)

    def test_json_feed_updated_timestamp_is_rejected(self):
        quote = '"updated": "2025-03-12"'
        page = quote + "\nG-Group signed a contract."
        _, reason = verify_news(
            news_candidate(
                quote=quote,
                published_date="2025-03-12",
            ),
            page_text=page,
        )
        self.assertIn("modified or updated timestamp", reason)

    def test_navigation_label_is_not_news(self):
        _, reason = verify_news(
            news_candidate(
                value="News",
                quote="Nyheter\n12. mars 2025",
            )
        )
        self.assertIn("navigation or section label", reason)

    def test_product_page_is_not_news(self):
        _, reason = verify_news(
            news_candidate(),
            source_url="https://example.no/products/widget",
        )
        self.assertIn("product or listing page", reason)

    def test_english_month_is_matched_in_the_quote(self):
        page = "G-Group signed a contract.\n14 October 2025"
        _, reason = verify_news(
            news_candidate(
                value="G-Group signed a contract.",
                quote="G-Group signed a contract.\n14 October 2025",
                published_date="2025-10-14",
            ),
            page_text=page,
        )
        self.assertIsNone(reason)

    def test_month_first_candidate_date_is_normalised(self):
        page = "G-Group signed a contract.\nOctober 14, 2025"
        fact, reason = verify_news(
            news_candidate(
                value="G-Group signed a contract.",
                quote="G-Group signed a contract.\nOctober 14, 2025",
                published_date="October 14, 2025",
            ),
            page_text=page,
        )
        self.assertIsNone(reason)
        self.assertEqual(fact.published_date, "2025-10-14")


PAGE_TWO = """
Vi leverer programvare for regnskap og lønn.
Selskapet ble stiftet i 2014 og har kontorer i Bergen.
"""


def page_input(
    text=PAGE,
    url="https://example.no/about",
    tier=2,
    title="About",
):
    return PageInput(
        page_text=text,
        source_url=url,
        retrieved_at="2026-10-13T00:00:00+00:00",
        tier=tier,
        source_type=(
            "official_company_website" if tier <= 2 else "business_directory"
        ),
        title=title,
    )


def batch_candidate(field, value, quote, page):
    return {
        "field": field,
        "value": value,
        "quote": quote,
        "published_date": "",
        "page": page,
    }


class BatchExtractionTests(unittest.TestCase):
    """Two pages share one call without sharing their evidence."""

    def _run(self, pages, facts):
        calls = []

        def model(schema, *, system, user):
            calls.append({"schema": schema, "system": system, "user": user})
            return llm.LLMResult(
                data={"facts": facts},
                model="m",
                provider="openai_compatible",
                finish_reason="stop",
            )

        with patch("src.extract.llm.is_available", return_value=True):
            with patch("src.extract.llm.complete_json", side_effect=model):
                reports = extract_from_pages(
                    pages,
                    company_number="989061593",
                    company_name="-G- GROUP AS",
                    registry_values=["-G- GROUP AS"],
                )
        return reports, calls

    def test_two_pages_share_one_call(self):
        reports, calls = self._run(
            [
                page_input(),
                page_input(PAGE_TWO, "https://example.no/tjenester"),
            ],
            [
                batch_candidate(
                    "business_summary",
                    "Investment company",
                    "We are a Norwegian investment company founded in 2005.",
                    "p1",
                ),
                batch_candidate(
                    "services",
                    "Accounting software",
                    "Vi leverer programvare for regnskap og lønn.",
                    "p2",
                ),
            ],
        )

        self.assertEqual(len(calls), 1)
        self.assertTrue(all(report.llm_used for report in reports))
        self.assertEqual([len(report.facts) for report in reports], [1, 1])

    def test_a_page_the_batch_skipped_is_read_alone(self):
        calls = []

        def model(schema, *, system, user):
            calls.append(user)
            facts = [
                batch_candidate(
                    "business_summary",
                    "Investment company",
                    "We are a Norwegian investment company founded in 2005.",
                    "p1",
                )
            ]
            if "=== PAGE p1 ===" not in user:
                facts = [
                    batch_candidate(
                        "services",
                        "Accounting software",
                        "Vi leverer programvare for regnskap og lønn.",
                        "p1",
                    )
                ]
            return llm.LLMResult(
                data={"facts": facts},
                model="m",
                provider="openai_compatible",
                finish_reason="stop",
            )

        with patch("src.extract.llm.is_available", return_value=True):
            with patch("src.extract.llm.complete_json", side_effect=model):
                reports = extract_from_pages(
                    [
                        page_input(),
                        page_input(PAGE_TWO, "https://example.no/tjenester"),
                    ],
                    company_number="989061593",
                    company_name="-G- GROUP AS",
                    registry_values=["-G- GROUP AS"],
                )

        self.assertEqual(len(calls), 2)
        self.assertEqual(
            [len(report.facts) for report in reports], [1, 1]
        )
        self.assertTrue(reports[1].retried_for_breadth)
        self.assertTrue(reports[1].facts[0].source_url.endswith("/tjenester"))

    def test_each_fact_is_attributed_to_its_own_page(self):
        reports, _ = self._run(
            [
                page_input(),
                page_input(PAGE_TWO, "https://example.no/tjenester"),
            ],
            [
                batch_candidate(
                    "business_summary",
                    "Investment company",
                    "We are a Norwegian investment company founded in 2005.",
                    "p1",
                ),
                batch_candidate(
                    "services",
                    "Accounting software",
                    "Vi leverer programvare for regnskap og lønn.",
                    "p2",
                ),
            ],
        )

        self.assertEqual(len(reports[0].facts), 1)
        self.assertEqual(len(reports[1].facts), 1)
        self.assertEqual(
            reports[0].facts[0].source_url, "https://example.no/about"
        )
        self.assertEqual(
            reports[1].facts[0].source_url,
            "https://example.no/tjenester",
        )

    def test_a_wrong_page_label_is_corrected_by_the_quote(self):
        reports, _ = self._run(
            [
                page_input(),
                page_input(PAGE_TWO, "https://example.no/tjenester"),
            ],
            [
                batch_candidate(
                    "business_summary",
                    "Investment company",
                    "We are a Norwegian investment company founded in 2005.",
                    "p2",
                )
            ],
        )

        self.assertEqual(len(reports[0].facts), 1)
        self.assertEqual(reports[1].facts, [])
        self.assertEqual(
            reports[0].facts[0].source_url, "https://example.no/about"
        )

    def test_a_quote_in_no_page_is_refused_against_the_page_it_names(self):
        reports, _ = self._run(
            [
                page_input(),
                page_input(PAGE_TWO, "https://example.no/tjenester"),
            ],
            [
                batch_candidate(
                    "business_summary",
                    "Forty offices",
                    "We run forty offices across Scandinavia.",
                    "p2",
                )
            ],
        )

        self.assertEqual(reports[0].facts, [])
        self.assertEqual(reports[1].facts, [])
        self.assertEqual(len(reports[1].rejected), 1)
        self.assertEqual(
            reports[1].rejected[0]["reason"], "quote not found in page"
        )

    def test_a_quote_in_both_pages_uses_the_named_page(self):
        shared = "\nVi har kontorer i Norge.\n"
        reports, _ = self._run(
            [
                page_input(PAGE + shared, "https://example.no/a"),
                page_input(PAGE_TWO + shared, "https://example.no/b"),
            ],
            [
                batch_candidate(
                    "business_summary",
                    "Norwegian offices",
                    "Vi har kontorer i Norge.",
                    "p2",
                )
            ],
        )

        self.assertEqual(reports[0].facts, [])
        self.assertEqual(len(reports[1].facts), 1)

    def test_the_same_statement_from_both_pages_is_published_once(self):
        shared = "\nVi har kontorer i Norge.\n"
        reports, _ = self._run(
            [
                page_input(PAGE + shared, "https://example.no/a"),
                page_input(PAGE_TWO + shared, "https://example.no/b"),
            ],
            [
                batch_candidate(
                    "business_summary",
                    "Norwegian offices",
                    "Vi har kontorer i Norge.",
                    "p1",
                ),
                batch_candidate(
                    "business_summary",
                    "Norwegian offices",
                    "Vi har kontorer i Norge.",
                    "p2",
                ),
            ],
        )

        self.assertEqual(len(reports[0].facts), 1)
        self.assertEqual(reports[1].facts, [])
        self.assertEqual(
            reports[1].rejected[0]["reason"],
            "duplicates an accepted business_summary fact",
        )

    def test_an_empty_page_is_reported_without_a_call(self):
        reports, calls = self._run(
            [
                page_input("   ", "https://example.no/empty"),
                page_input(PAGE_TWO, "https://example.no/tjenester"),
            ],
            [
                batch_candidate(
                    "services",
                    "Accounting software",
                    "Vi leverer programvare for regnskap og lønn.",
                    "p1",
                )
            ],
        )

        self.assertEqual(len(calls), 1)
        self.assertIn("no text content", reports[0].error)
        self.assertEqual(len(reports[1].facts), 1)

    def test_an_oversized_page_falls_back_to_single_reads(self):
        huge = PAGE + ("x" * (MAX_BATCH_PAGE_CHARS + 1))
        reports, calls = self._run(
            [
                page_input(),
                page_input(huge, "https://example.no/long"),
            ],
            [],
        )

        self.assertEqual(len(calls), 2)
        self.assertTrue(all(report.llm_used for report in reports))

    def test_a_truncated_batch_falls_back_to_single_reads(self):
        calls = []

        def model(schema, *, system, user):
            calls.append(user)
            if len(calls) == 1:
                raise llm.LLMTruncated("response was cut off")
            return llm.LLMResult(
                data={"facts": []},
                model="m",
                provider="openai_compatible",
                finish_reason="stop",
            )

        with patch("src.extract.llm.is_available", return_value=True):
            with patch("src.extract.llm.complete_json", side_effect=model):
                reports = extract_from_pages(
                    [
                        page_input(),
                        page_input(PAGE_TWO, "https://example.no/tjenester"),
                    ],
                    company_number="989061593",
                    company_name="-G- GROUP AS",
                    registry_values=["-G- GROUP AS"],
                )

        self.assertEqual(len(calls), 3)
        self.assertTrue(all(report.llm_used for report in reports))
        self.assertIsNone(reports[0].error)
        self.assertIsNone(reports[1].error)

    def test_a_rate_limited_batch_is_not_retried_once_per_page(self):
        calls = []

        def model(*_args, **_kwargs):
            calls.append(1)
            raise llm.LLMCallFailed("rate limited (429)")

        with patch("src.extract.llm.is_available", return_value=True):
            with patch("src.extract.llm.complete_json", side_effect=model):
                reports = extract_from_pages(
                    [
                        page_input(),
                        page_input(PAGE_TWO, "https://example.no/tjenester"),
                    ],
                    company_number="989061593",
                    company_name="-G- GROUP AS",
                    registry_values=["-G- GROUP AS"],
                )

        self.assertEqual(len(calls), 1)
        self.assertTrue(all("429" in (report.error or "") for report in reports))

    def test_a_restatement_only_batch_page_still_gets_its_second_pass(self):
        calls = []

        def model(schema, *, system, user):
            is_batch = "page" in schema["properties"]["facts"]["items"][
                "properties"
            ]
            calls.append({"schema": schema, "user": user})
            if is_batch:
                return llm.LLMResult(
                    data={
                        "facts": [
                            candidate(value="-G- GROUP AS",
                                      quote="- G- GROUP AS")
                        ]
                    },
                    model="m",
                    provider="openai_compatible",
                    finish_reason="stop",
                )
            return llm.LLMResult(
                data={"facts": [candidate()]},
                model="m",
                provider="openai_compatible",
                finish_reason="stop",
            )

        with patch("src.extract.llm.is_available", return_value=True):
            with patch("src.extract.llm.complete_json", side_effect=model):
                reports = extract_from_pages(
                    [
                        page_input(),
                        page_input(PAGE_TWO, "https://example.no/tjenester"),
                    ],
                    company_number="989061593",
                    company_name="-G- GROUP AS",
                    registry_values=["-G- GROUP AS"],
                )

        self.assertTrue(reports[0].retried_for_breadth)
        self.assertEqual(len(reports[0].facts), 1)
        self.assertEqual(len(calls), 2)

    def test_batch_prompt_and_schema_require_page_attribution(self):
        _, calls = self._run(
            [
                page_input(),
                page_input(PAGE_TWO, "https://example.no/tjenester"),
            ],
            [],
        )

        items = calls[0]["schema"]["properties"]["facts"]["items"]
        self.assertIn("page", items["required"])
        self.assertEqual(items["properties"]["page"]["enum"], ["p1", "p2"])
        self.assertIn("PAGE p1", calls[0]["user"])
        self.assertIn("PAGE p2", calls[0]["user"])
        self.assertIn("`page`", BATCH_SYSTEM_PROMPT)
        # Building a batch schema must not mutate the shared one.
        self.assertNotIn(
            "page", EXTRACTION_SCHEMA["properties"]["facts"]["items"]["properties"]
        )


class BreadthRetryTests(unittest.TestCase):
    """The second pass is a model call, so it has to earn its place.

    A directory that only restated the registry is not worth a second
    call once the company already has facts: measured across fifty
    companies, directory pages yielded 0.11 facts each.
    """

    def _run(self, *, tier, seen, calls):
        restatement = {
            "facts": [candidate(value="-G- GROUP AS", quote="- G- GROUP AS")],
        }
        useful = {
            "facts": [
                candidate(
                    value="investment company",
                    quote="We are a Norwegian investment company founded in 2005.",
                )
            ],
        }

        def model(*_args, **_kwargs):
            calls.append(_kwargs.get("user") or (_args[2] if _args else ""))
            return llm.LLMResult(
                data=restatement if len(calls) == 1 else useful,
                model="m",
                provider="anthropic",
                finish_reason="end_turn",
            )

        with patch("src.extract.llm.is_available", return_value=True):
            with patch("src.extract.llm.complete_json", side_effect=model):
                return extract_from_page(
                    page_text=PAGE,
                    source_url="https://example.no/about",
                    retrieved_at="2026-10-13T00:00:00+00:00",
                    company_number="989061593",
                    company_name="-G- GROUP AS",
                    tier=tier,
                    source_type=(
                        "official_company_website"
                        if tier <= 2
                        else "business_directory"
                    ),
                    registry_values=["-G- GROUP AS"],
                    already_accepted=seen,
                )

    @staticmethod
    def _fact():
        from src.extract import ExtractedFact

        return ExtractedFact(
            field="business_summary",
            value="something already published",
            quote="Oslo",
            source_url="https://example.no/about",
            retrieved_at="2026-10-13T00:00:00+00:00",
            tier=2,
            source_type="official_company_website",
            confidence=0.9,
        )

    def test_first_party_page_still_gets_a_second_pass(self):
        calls = []
        report = self._run(tier=2, seen=[self._fact()], calls=calls)

        self.assertEqual(len(calls), 2)
        self.assertTrue(report.retried_for_breadth)

    def test_directory_page_is_not_retried_once_facts_exist(self):
        calls = []
        report = self._run(tier=6, seen=[self._fact()], calls=calls)

        self.assertEqual(len(calls), 1)
        self.assertFalse(report.retried_for_breadth)

    def test_directory_still_gets_a_second_pass_when_the_company_has_nothing(self):
        calls = []
        report = self._run(tier=6, seen=[], calls=calls)

        self.assertEqual(len(calls), 2)
        self.assertTrue(report.retried_for_breadth)


if __name__ == "__main__":
    unittest.main()
