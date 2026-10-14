"""
Personal-content refusal.

The prompt tells the model not to report biography. This tests whether
the code refuses it anyway, because that is the actual guarantee: a
pipeline whose one defence against biography is an instruction will
eventually publish someone's medical history.

The hard cases are the false positives. "Family-owned", "healthcare",
"children's products" and a named spouse in a business sentence must all
survive. A filter that eats real company facts is worse than no filter,
because it looks like working.
"""

import unittest

from src.extract import _is_personal_content, verify_candidate

PAGE = "Page content about a company."


class PersonalSignalTests(unittest.TestCase):
    def test_plain_business_content_passes(self):
        for text in (
            "The company provides reliable energy for a world in transition.",
            "Headquartered in Norway with offices in more than 20 countries.",
            "A major supplier of energy to Europe.",
        ):
            self.assertIsNone(_is_personal_content(text, text), text)

    def test_care_business_terms_are_not_personal(self):
        # These are the false positives that matter most: each contains a
        # word from a plausible biography lexicon, in a business context.
        for text in (
            "A family-owned Norwegian manufacturer.",
            "Family businesses have been our focus since 1920.",
            "Healthcare services for hospitals and clinics.",
            "Health and safety are core to our operations.",
            "We make products for children's hospitals.",
            "Child labour is prohibited in our supply chain.",
            "Best in class solutions for your business.",
            "Hobbies are our business, and we supply them nationwide.",
        ):
            self.assertIsNone(_is_personal_content(text, text), text)

    def test_pure_biography_is_refused(self):
        cases = [
            "He is married to Anna and they have three children.",
            "She was born in Bergen and grew up in Trondheim.",
            "Our founder's wife Anna led the charity until her death.",
            "He passed away in 2021; she survived by three children.",
        ]
        for text in cases:
            self.assertIsNotNone(_is_personal_content(text, text), text)

    def test_single_incidental_mention_survives(self):
        # One spouse named inside a long business sentence is not a
        # biography page.
        text = (
            "The company was founded by Karl Berg in 1901 and is today led "
            "by Ingrid Holm, whose wife serves on the local school board. "
            "It supplies industrial valves across Scandinavia and exports "
            "to more than thirty countries."
        )
        self.assertIsNone(_is_personal_content(text, text))

    def test_refusal_is_scored_on_the_quote(self):
        # The fact's value may be innocuous while its quote is biography.
        self.assertIsNotNone(
            _is_personal_content(
                "Company history",
                "He was born in 1952 and his wife Anna ran the local shop.",
            )
        )

    def test_falls_back_to_value_when_no_quote(self):
        self.assertIsNotNone(
            _is_personal_content("His wife Anna runs the local shop.", "")
        )

    def test_empty_input_is_not_personal(self):
        self.assertIsNone(_is_personal_content("", ""))


class PersonalRefusalIntegrationTests(unittest.TestCase):
    def _verify(self, quote, value=None):
        return verify_candidate(
            {
                "field": "business_summary",
                "value": value or quote,
                "quote": quote,
            },
            page_text=PAGE + " " + quote,
            source_url="https://example.no/about",
            retrieved_at="2026-01-01T00:00:00Z",
            tier=2,
            source_type="first_party",
            company_number="923609016",
        )

    def test_biography_fact_is_refused(self):
        fact, reason = self._verify(
            "He is married to Anna and they have three children."
        )
        self.assertIsNone(fact)
        self.assertIsNotNone(reason)
        self.assertIn("biography", reason)

    def test_business_fact_is_accepted(self):
        fact, reason = self._verify(
            "A major supplier of energy to Europe."
        )
        self.assertIsNone(reason)
        self.assertIsNotNone(fact)

    def test_refusal_happens_before_deduplication(self):
        # The recorded reason must be the substantive one, not "duplicate",
        # or the audit trail misleads whoever reads it.
        fact, reason = self._verify(
            "He is married to Anna and they have three children."
        )
        self.assertNotIn("duplicate", reason or "")


if __name__ == "__main__":
    unittest.main()
