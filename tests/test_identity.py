"""Identity verification — the wrong-company guard.

The cases here come from real TinyFish results captured against live
Norwegian organizations, not from invented examples.
"""

import json
import unittest
from pathlib import Path

from src.identity import (
    CompanyIdentity,
    get_domain,
    name_tokens,
    strip_legal_form,
    verify,
)

SAMPLE = Path(__file__).resolve().parent / "fixtures" / "search_sample.json"


def identity(**overrides) -> CompanyIdentity:
    base = {
        "company_number": "989061593",
        "name": "-G- GROUP AS",
        "address": "c/o Gjørva, Hellerud gårdsvei 5B, 0671 OSLO, Norge",
        "postal_code": "0671",
        "registered_website": None,
        "registered_domain": None,
    }
    base.update(overrides)
    return CompanyIdentity(**base)


def result(url, title="", snippet=""):
    return {"url": url, "title": title, "snippet": snippet}


class NameNormalisationTests(unittest.TestCase):
    def test_strips_legal_form(self):
        self.assertEqual(strip_legal_form("Equinor ASA"), "equinor")
        self.assertEqual(strip_legal_form("-G- GROUP AS"), "g group")
        # A legal form is only stripped as a trailing token; a company
        # whose name merely starts with one keeps it.
        self.assertEqual(
            strip_legal_form("Aksjeselskapet Xylophon AS"),
            "aksjeselskapet xylophon",
        )

    def test_folds_norwegian_characters_to_ascii(self):
        # The fold must agree with how these names are typed in ASCII by
        # sources that cannot represent the Norwegian letters.
        self.assertEqual(strip_legal_form("Gjørva AS"), "gjorva")
        self.assertEqual(strip_legal_form("Blæder AS"), "blaeder")
        self.assertEqual(
            strip_legal_form("Hellebø-Storeide AS"), "hellebo storeide"
        )

    def test_drops_short_and_generic_tokens(self):
        # A one- or two-character fragment cannot confirm identity.
        tokens = name_tokens("AB AS")
        self.assertNotIn("ab", tokens)
        self.assertIn("xyzzy", name_tokens("Xyzzy AS"))


class RejectionTests(unittest.TestCase):
    def test_rejects_missing_url(self):
        verdict = verify(result(""), identity())
        self.assertFalse(verdict.verified)

    def test_rejects_person_page(self):
        verdict = verify(
            result("https://firmadatabasen.no/person/geir-inge-gjorva-1967"),
            identity(),
        )
        self.assertFalse(verdict.verified)
        self.assertEqual(verdict.basis, "person_page")

    def test_rejects_listing_page_carrying_the_org_number(self):
        # The org number is present, but it identifies the query rather
        # than any single company listed on the page.
        verdict = verify(
            result(
                "https://www.proff.no/bransjes%C3%B8k?q=989061593",
                title='"989061593" i bedrifter. 1 treff',
            ),
            identity(),
        )
        self.assertFalse(verdict.verified)
        self.assertEqual(verdict.basis, "listing_page")

    def test_rejects_explicit_not_found(self):
        verdict = verify(
            result("https://example.no/989061593", snippet="Foretaket finnes ikke"),
            identity(),
        )
        self.assertFalse(verdict.verified)

    def test_aggregator_name_match_alone_is_insufficient(self):
        # Real case: nor47business and sceneweb pages whose URL carries no
        # org number at all.
        for url in (
            "https://www.nor47business.com/nb/company/Ggroup-As",
            "https://sceneweb.no/nb/venue/160590/Palmera",
        ):
            verdict = verify(
                result(url, title="-G- GROUP AS", snippet="Eksempelveien 1"),
                identity(),
            )
            self.assertFalse(verdict.verified, url)
            self.assertTrue(
                any("aggregator" in r for r in verdict.reasons), url
            )

    def test_unrelated_domain_with_only_a_name_is_rejected(self):
        verdict = verify(
            result("https://some-blog.example/x", title="-G- GROUP AS"),
            identity(),
        )
        self.assertFalse(verdict.verified)


class AcceptanceTests(unittest.TestCase):
    def test_org_number_in_url_path_accepts(self):
        verdict = verify(
            result("https://bedri.no/bedrift/989061593"),
            identity(),
        )
        self.assertTrue(verdict.verified)
        self.assertEqual(verdict.basis, "organization_number")

    def test_org_number_on_page_accepts(self):
        verdict = verify(
            result("https://example.no/x", snippet="Org.nr 989 061 593"),
            identity(),
        )
        self.assertTrue(verdict.verified)

    def test_registry_lookup_parameter_accepts(self):
        verdict = verify(
            result(
                "https://w2.brreg.no/kunngjoring/hent_nr.jsp?orgnr=989061593",
                snippet="Ingen informasjon er tilgjengelig for denne siden.",
            ),
            identity(),
        )
        self.assertTrue(verdict.verified)

    def test_registered_website_plus_name_accepts(self):
        # Real case: the ministry's own site carries no org number in its
        # URL, but the domain is the registered one.
        ident = identity(
            company_number="983887457",
            name="ARBEIDS- OG INKLUDERINGSDEPARTEMENTET",
            registered_website="regjeringen.no/aid",
            registered_domain="regjeringen.no",
        )
        verdict = verify(
            result(
                "https://www.regjeringen.no/no/dep/aid/kontakt/id173/",
                title="Arbeids- og inkluderingsdepartementet",
                snippet="Akersgata 64",
            ),
            ident,
        )
        self.assertTrue(verdict.verified)
        self.assertEqual(verdict.basis, "registered_domain_and_name")

    def test_foreign_org_number_is_recorded_as_a_reason(self):
        verdict = verify(
            result(
                "https://example.no/x",
                snippet="Org.nr 989061593 og 983887457",
            ),
            identity(),
        )
        self.assertTrue(verdict.verified)
        self.assertIn("983887457", " ".join(verdict.reasons))


class DomainNormalisationTests(unittest.TestCase):
    """Brreg stores websites as bare hostnames, not URLs."""

    def test_bare_hostname_yields_a_domain(self):
        # urlparse reads a scheme-less string as a path, so netloc was
        # empty and registered_domain was None for every real company.
        self.assertEqual(get_domain("www.equinor.com"), "equinor.com")

    def test_full_url_and_bare_hostname_agree(self):
        self.assertEqual(
            get_domain("https://www.equinor.com"),
            get_domain("www.equinor.com"),
        )

    def test_path_is_discarded(self):
        self.assertEqual(get_domain("https://m.example.no/a/b"), "example.no")

    def test_empty_inputs_are_safe(self):
        self.assertEqual(get_domain(""), "")
        self.assertEqual(get_domain(None), "")

    def test_credentials_are_not_part_of_the_domain(self):
        self.assertEqual(
            get_domain("https://user:pw@shop.example.no/a"), "shop.example.no"
        )

    def test_registered_domain_is_populated_from_brreg_shape(self):
        parsed = CompanyIdentity.from_brreg(
            {"company_number": "923609016", "company_name": "EQUINOR ASA",
             "website": "www.equinor.com"}
        )
        self.assertEqual(parsed.registered_domain, "equinor.com")

    def test_missing_website_leaves_it_absent(self):
        parsed = CompanyIdentity.from_brreg(
            {"company_number": "923609016", "company_name": "X AS"}
        )
        self.assertIsNone(parsed.registered_domain)


class SparseIdentityTests(unittest.TestCase):
    """An entity with no registered address must still verify cleanly."""

    def test_verification_survives_a_missing_postal_code(self):
        # postal_code_match used to be indexed unconditionally in the
        # scoring dict while only being set when the registry held a
        # postal code, so every address-less company raised KeyError.
        sparse = identity(
            company_number="923609016",
            name="EQUINOR ASA",
            address=None,
            postal_code=None,
        )
        verdict = verify(
            {
                "url": "https://bedri.no/bedrift/923609016",
                "title": "Equinor",
                "snippet": "Org.nr 923609016",
            },
            sparse,
        )
        self.assertTrue(verdict.verified)
        self.assertIn("postal_code_match", verdict.signals)
        self.assertFalse(verdict.signals["postal_code_match"])

    def test_every_signal_key_is_always_present(self):
        sparse = identity(address=None, postal_code=None)
        verdict = verify(
            {"url": "https://example.no/x", "title": "-G- GROUP AS"},
            sparse,
        )
        for key in (
            "org_number_in_url_path",
            "org_number_in_registry_query",
            "org_number_on_page",
            "registered_domain_match",
            "exact_name_in_title",
            "exact_name_on_page",
            "name_token_overlap",
            "registered_path_token_match",
            "previous_name_on_page",
            "postal_code_match",
        ):
            self.assertIn(key, verdict.signals, key)

    def test_postal_code_still_counts_when_known(self):
        full = identity(postal_code="0671")
        verdict = verify(
            {
                "url": "https://bedri.no/bedrift/989061593",
                "snippet": "0671 Oslo",
            },
            full,
        )
        self.assertTrue(verdict.signals["postal_code_match"])


class RealSearchResultTests(unittest.TestCase):
    """Replay the captured corpus end to end."""

    @classmethod
    def setUpClass(cls):
        if not SAMPLE.exists():
            raise unittest.SkipTest("search sample not captured")
        cls.corpus = json.loads(SAMPLE.read_text(encoding="utf-8"))

    def test_listing_and_person_pages_never_accepted(self):
        identities = {
            "989061593": identity(),
            "983887457": identity(
                company_number="983887457",
                name="ARBEIDS- OG INKLUDERINGSDEPARTEMENTET",
                registered_website="regjeringen.no/aid",
                registered_domain="regjeringen.no",
            ),
        }
        for org, results in self.corpus.items():
            ident = identities.get(org, identity(company_number=org))
            for item in results:
                verdict = verify(item, ident)
                if verdict.classification and (
                    verdict.classification.is_listing_page
                    or verdict.classification.is_person_page
                ):
                    self.assertFalse(
                        verdict.verified,
                        f"{item['url']} should never be accepted",
                    )


if __name__ == "__main__":
    unittest.main()
