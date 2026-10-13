"""Source tiering and URL structural classification."""

import unittest

from src.sources import (
    TIER_DIRECTORY,
    TIER_OFFICIAL_REGISTRY,
    TIER_UNVERIFIED,
    classify,
    confidence_for,
    find_org_number,
    org_numbers_in,
)


class OrgNumberExtractionTests(unittest.TestCase):
    def test_reads_plain_number(self):
        self.assertEqual(find_org_number("Org.nr: 989061593"), "989061593")

    def test_reads_registry_separators(self):
        self.assertEqual(find_org_number("983 887 457"), "983887457")
        self.assertEqual(find_org_number("983-887-457"), "983887457")

    def test_rejects_digits_inside_a_longer_run(self):
        # A 12-digit run must not yield a match.
        self.assertIsNone(find_org_number("123456789012"))

    def test_finds_all_in_text(self):
        numbers = org_numbers_in("A: 989061593 and B: 983 887 457")
        self.assertEqual(numbers, {"989061593", "983887457"})


class TierAssignmentTests(unittest.TestCase):
    def test_government_registry_is_tier_one(self):
        for url in (
            "https://data.brreg.no/enhetsregisteret/api/enheter/1",
            "https://virksomhet.brreg.no/nb/oppslag/enheter/1",
            "https://www.regjeringen.no/no/dep/aid/kontakt/id173/",
            "https://forvaltningsdatabasen.sikt.no/data/organisasjon/1",
        ):
            self.assertEqual(classify(url).tier, TIER_OFFICIAL_REGISTRY, url)

    def test_business_database_is_tier_four(self):
        result = classify("https://opplysning.byndle.no/selskap/x/1")
        self.assertEqual(result.tier, 4)

    def test_aggregator_is_directory_tier(self):
        result = classify("https://purehelp.no/m/company/details/x/1")
        self.assertEqual(result.tier, TIER_DIRECTORY)
        self.assertTrue(result.is_aggregator)

    def test_unknown_domain_is_unverified(self):
        result = classify("https://some-random-blog.example/thing")
        self.assertEqual(result.tier, TIER_UNVERIFIED)


class ListingPageDetectionTests(unittest.TestCase):
    def test_search_query_page_is_a_listing(self):
        # This is the page that produced a phone number belonging to an
        # unrelated company before the listing check existed.
        result = classify(
            "https://www.proff.no/bransjes%C3%B8k?q=989061593", "989061593"
        )
        self.assertTrue(result.is_listing_page)
        self.assertEqual(result.tier, TIER_UNVERIFIED)

    def test_browse_page_is_a_listing(self):
        result = classify("https://forvalt.no/Nettbutikk/produkter/922936919", "922936919")
        self.assertTrue(result.is_listing_page)

    def test_registry_lookup_parameter_is_not_a_listing(self):
        result = classify(
            "https://w2.brreg.no/kunngjoring/hent_nr.jsp?orgnr=989061593",
            "989061593",
        )
        self.assertFalse(result.is_listing_page)
        self.assertEqual(result.tier, TIER_OFFICIAL_REGISTRY)
        self.assertEqual(result.org_number_in_query, "989061593")


class PersonPageDetectionTests(unittest.TestCase):
    def test_person_record_is_flagged(self):
        result = classify("https://firmadatabasen.no/person/geir-inge-gjorva-1967")
        self.assertTrue(result.is_person_page)


class ForeignIdentifierTests(unittest.TestCase):
    """Directory IDs are 9 digits too and must not pass as org numbers."""

    def test_own_directory_id_is_not_the_target(self):
        result = classify(
            "https://www.gulesider.no/-palmera/125157773/bedrift", "916627939"
        )
        self.assertIsNone(result.org_number_in_path)
        self.assertIn("125157773", result.foreign_numbers)

    def test_target_alongside_another_number_still_matches(self):
        result = classify(
            "https://www.scribd.com/document/991205763/934551435-SIGNERT",
            "934551435",
        )
        self.assertEqual(result.org_number_in_path, "934551435")
        self.assertIn("991205763", result.foreign_numbers)


class ConfidenceTests(unittest.TestCase):
    def test_tier_ordering_is_strictly_descending(self):
        ceilings = [confidence_for(t) for t in range(1, 8)]
        self.assertEqual(ceilings, sorted(ceilings, reverse=True))

    def test_registry_is_certain(self):
        self.assertEqual(confidence_for(1), 1.0)

    def test_corroboration_cannot_exceed_registry(self):
        self.assertEqual(confidence_for(1, corroborated=True), 1.0)

    def test_corroboration_lifts_a_lower_tier_one_step(self):
        self.assertEqual(
            confidence_for(4, corroborated=True), confidence_for(3)
        )

    def test_unknown_tier_falls_back_to_unverified(self):
        self.assertEqual(confidence_for(99), confidence_for(7))


if __name__ == "__main__":
    unittest.main()
