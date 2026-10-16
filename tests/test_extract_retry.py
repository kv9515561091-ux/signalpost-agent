"""
Retry-for-breadth.

The measured failure this addresses: sixteen of the twenty companies
yielding no facts at all had every proposal refused for "repeats a value
already sourced from the registry". For a single-product company whose
website says what Brreg says, that refusal is usually correct. But it is
also what a page that *does* describe products and customers looks like
when the model stops at the first field it recognises.

So a second attempt is made, told what was rejected and why. It costs a
model call and no additional request.

The risk that decides whether this is safe: a retry that relaxes the
registry check would let the agent publish, as a discovery, exactly the
fields the registry already holds. Coverage bought that way is
fabrication, which disqualifies the entry outright.
"""

import unittest
from unittest.mock import patch

from src import extract as extract_module
from src.extract import extract_from_page

PAGE = """
Karlstad Bruk AS. Org.nr 976543210.
Vi produserer og leverer industrielle ventilasjonssystemer til
byggebransjen i Norge og Sverige. Vi har 45 ansatte.
Vår adresse er Storgata 1, 0150 Oslo.
"""

REGISTRY_VALUES = ("KARLSTAD BRUK AS", "Storgata 1, 0150 OSLO")


def _result(facts):
    return type(
        "Result",
        (),
        {
            "data": {"facts": facts},
            "model": "stub",
            "provider": "stub",
            "finish_reason": "stop",
            "input_tokens": 0,
            "output_tokens": 0,
        },
    )()


def _run(passes):
    """Feed a sequence of model answers, one per call."""
    calls = iter(passes)

    def fake_complete_json(schema, *, system, user, **kwargs):
        return next(calls)

    with patch("src.extract.llm.is_available", return_value=True), patch(
        "src.extract.llm.unavailable_reason", return_value=None
    ), patch("src.extract.llm.complete_json", fake_complete_json):
        return extract_from_page(
            page_text=PAGE,
            source_url="https://karlstadbru.no",
            retrieved_at="2026-01-01T00:00:00Z",
            company_number="976543210",
            company_name="KARLSTAD BRUK AS",
            tier=2,
            source_type="first_party",
            registry_values=REGISTRY_VALUES,
        )


RESTATEMENT = {
    "field": "business_summary",
    "value": "KARLSTAD BRUK AS",
    "quote": "Karlstad Bruk AS. Org.nr 976543210.",
    "published_date": None,
}

REAL_FACT = {
    "field": "products",
    "value": "We produce and supply industrial ventilation systems",
    "quote": "Vi produserer og leverer industrielle ventilasjonssystemer",
    "published_date": None,
}


class RetryTriggerTests(unittest.TestCase):
    def test_no_retry_when_the_first_pass_succeeds(self):
        report = _run([_result([REAL_FACT])])
        self.assertEqual(len(report.facts), 1)
        self.assertFalse(report.retried_for_breadth)

    def test_retry_after_pure_registry_restatement(self):
        report = _run([_result([RESTATEMENT]), _result([REAL_FACT])])

        self.assertTrue(report.retried_for_breadth)
        self.assertEqual(len(report.facts), 1, "the retry found nothing useful")
        self.assertEqual(report.facts[0].field, "products")

    def test_no_retry_when_nothing_was_rejected_at_all(self):
        # An empty first pass is a real answer, not a misfire.
        report = _run([_result([])])
        self.assertFalse(report.retried_for_breadth)

    def test_no_retry_when_refusals_were_mixed(self):
        # A page that also had a bad quote is being read fine.
        report = _run(
            [
                _result(
                    [
                        RESTATEMENT,
                        {
                            "field": "products",
                            "value": "invented",
                            "quote": "text that is not on the page",
                            "published_date": None,
                        },
                    ]
                )
            ]
        )
        self.assertFalse(report.retried_for_breadth)

    def test_retry_does_not_repeat_the_first_pass_refusals(self):
        report = _run([_result([RESTATEMENT]), _result([REAL_FACT])])

        # The first pass's refusal is kept separately rather than
        # concatenated, so the recorded rejection list describes the
        # attempt actually in force.
        self.assertEqual(len(report.retried_rejections), 1)
        self.assertEqual(report.rejected, [])


class SafetyTests(unittest.TestCase):
    """The retry must not become a way to publish registry fields."""

    def test_retry_cannot_publish_a_registry_restatement(self):
        # The second pass proposes only the registry name again. The
        # registry check is not relaxed by the retry.
        report = _run([_result([RESTATEMENT]), _result([RESTATEMENT])])

        self.assertEqual(report.facts, [], "a restatement leaked through the retry")
        self.assertTrue(report.retried_for_breadth)

    def test_retry_cannot_publish_an_unverifiable_quote(self):
        fabricated = {
            "field": "products",
            "value": "We supply nuclear cooling systems",
            "quote": "We supply nuclear cooling systems",
            "published_date": None,
        }
        report = _run([_result([RESTATEMENT]), _result([fabricated])])

        self.assertEqual(report.facts, [], "an unquotable claim was published")
        self.assertTrue(report.rejected)

    def test_retry_cannot_publish_another_company(self):
        # The org-number check must survive the retry untouched.
        other = {
            "field": "business_summary",
            "value": "A different company",
            "quote": "A different company, org.nr 923609016",
            "published_date": None,
        }
        page_with_other = PAGE + "\nSee also org.nr 923609016 for details.\n"

        calls = iter([_result([RESTATEMENT]), _result([other])])
        with patch("src.extract.llm.is_available", return_value=True), patch(
            "src.extract.llm.unavailable_reason", return_value=None
        ), patch(
            "src.extract.llm.complete_json",
            lambda schema, *, system, user, **kw: next(calls),
        ):
            report = extract_from_page(
                page_text=page_with_other,
                source_url="https://karlstadbru.no",
                retrieved_at="2026-01-01T00:00:00Z",
                company_number="976543210",
                company_name="KARLSTAD BRUK AS",
                tier=2,
                source_type="first_party",
                registry_values=REGISTRY_VALUES,
            )

        self.assertEqual(report.facts, [], "another company's claim was published")


class PromptTests(unittest.TestCase):
    def test_breadth_hint_explains_the_rejection(self):
        hint = extract_module._BREADTH_HINT
        self.assertIn("registry", hint.lower())
        self.assertIn("empty", hint.lower())

    def test_restatements_only_detects_the_case(self):
        self.assertTrue(
            extract_module._restatements_only(
                [{"reason": "repeats a value already sourced from the registry"}]
            )
        )
        self.assertFalse(extract_module._restatements_only([]))
        self.assertFalse(
            extract_module._restatements_only([{"reason": "quote not found in page"}])
        )

    def test_restatements_only_requires_every_refusal_to_be_one(self):
        self.assertFalse(
            extract_module._restatements_only(
                [
                    {"reason": "repeats a value already sourced from the registry"},
                    {"reason": "quote not found in page"},
                ]
            )
        )


if __name__ == "__main__":
    unittest.main()
