"""Source tiering and URL structural classification."""

import unittest

from src.sources import (
    TIER_DIRECTORY,
    TIER_UNVERIFIED,
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


class RankingTests(unittest.TestCase):
    """Authority must come first: this decides what gets fetched."""

    def test_registry_ranks_before_directory(self):
        from src.sources import rank

        ordered = rank([
            {"url": "https://purehelp.no/x", "tier": 6},
            {"url": "https://data.brreg.no/x", "tier": 1},
            {"url": "https://example.no", "tier": 4},
        ])
        self.assertEqual(
            [s["tier"] for s in ordered], [1, 4, 6]
        )

    def test_tiers_are_ascending(self):
        # Tier numbers increase as authority decreases, so the output
        # must be non-decreasing. Sorting on -tier here silently reversed
        # the whole order and made extraction read the worst sources.
        from src.sources import rank

        ordered = rank([{"tier": t} for t in (7, 2, 5, 1, 4, 6, 3)])
        tiers = [s["tier"] for s in ordered]
        self.assertEqual(tiers, sorted(tiers))
        self.assertEqual(tiers[0], 1)
        self.assertEqual(tiers[-1], 7)

    def test_verified_precedes_unverified_within_a_tier(self):
        from src.sources import rank

        ordered = rank([
            {"url": "a", "tier": 4, "identity_verified": False},
            {"url": "b", "tier": 4, "identity_verified": True},
        ])
        self.assertEqual(ordered[0]["url"], "b")

    def test_ranking_is_stable(self):
        from src.sources import rank

        items = [{"url": "a", "tier": 3}, {"url": "b", "tier": 3}]
        self.assertEqual([s["url"] for s in rank(items)], ["a", "b"])

    def test_missing_tier_is_treated_as_least_authoritative(self):
        # Defaulting a missing tier to 7 means an unrecognised source can
        # never outrank a real one.
        from src.sources import rank

        ordered = rank([
            {"url": "unknown"},
            {"url": "official", "tier": 1},
        ])
        self.assertEqual(ordered[0]["url"], "official")
        self.assertEqual(ordered[-1]["url"], "unknown")

    def test_missing_tier_never_outranks_a_directory(self):
        from src.sources import rank

        ordered = rank([
            {"url": "unknown"},
            {"url": "directory", "tier": 6},
        ])
        self.assertEqual(ordered[0]["url"], "directory")


class MunicipalityListingTests(unittest.TestCase):
    """A page listing every company in a city is not about one company."""

    def test_municipality_index_is_a_listing(self):
        result = classify(
            "https://firmadatabasen.no/kommune/bergen/selskaper",
            "916627939",
        )
        self.assertTrue(result.is_listing_page)
        self.assertEqual(result.tier, TIER_UNVERIFIED)


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
