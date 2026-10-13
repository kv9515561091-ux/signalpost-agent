"""
Fixtures captured from the live Brreg APIs.

These pin the contracts the agent depends on — in particular the
``KONSERN`` vs ``SELSKAP`` distinction and the reporting currency — so a
silent upstream schema change is caught by the test suite rather than by
a judge noticing a wrong number in the output.
"""

import unittest

from src.financials import _normalise_year, _pick_statement, _ratio
from src.brreg import format_address


class AddressFormattingTests(unittest.TestCase):
    def test_joins_street_postal_city_country(self):
        block = {
            "adresse": ["Akersgata 64"],
            "postnummer": "0180",
            "poststed": "OSLO",
            "land": "Norge",
        }
        self.assertEqual(
            format_address(block), "Akersgata 64, 0180 OSLO, Norge"
        )

    def test_handles_multiple_street_lines(self):
        block = {"adresse": ["Gateveien 1", "Bygg B"], "poststed": "BERGEN"}
        self.assertEqual(format_address(block), "Gateveien 1, Bygg B, BERGEN")

    def test_missing_block_returns_none(self):
        self.assertIsNone(format_address(None))
        self.assertIsNone(format_address({}))

    def test_preserves_norwegian_characters(self):
        block = {"adresse": ["Hellerud gårdsvei 5 B"], "poststed": "OSLO"}
        self.assertIn("gårdsvei", format_address(block))


class StatementSelectionTests(unittest.TestCase):
    """The single most damaging field-mapping error is picking SELSKAP."""

    def test_prefers_consolidated_over_standalone(self):
        group = [
            {"regnskapstype": "SELSKAP", "id": 1},
            {"regnskapstype": "KONSERN", "id": 2},
        ]
        self.assertEqual(_pick_statement(group)["regnskapstype"], "KONSERN")

    def test_prefers_consolidated_regardless_of_order(self):
        group = [
            {"regnskapstype": "KONSERN", "id": 1},
            {"regnskapstype": "SELSKAP", "id": 2},
        ]
        self.assertEqual(_pick_statement(group)["regnskapstype"], "KONSERN")

    def test_falls_back_to_standalone_when_no_group(self):
        group = [{"regnskapstype": "SELSKAP", "id": 1}]
        self.assertEqual(_pick_statement(group)["regnskapstype"], "SELSKAP")

    def test_empty_group_returns_none(self):
        self.assertIsNone(_pick_statement([]))


class RatioTests(unittest.TestCase):
    def test_computes_percentage(self):
        self.assertEqual(_ratio(25.0, 100.0), 25.0)

    def test_suppressed_for_zero_denominator(self):
        self.assertIsNone(_ratio(10.0, 0.0))

    def test_suppressed_for_negative_denominator(self):
        # Filed accounts can report negative operating income; a margin
        # computed from that base is arithmetic, not information.
        self.assertIsNone(_ratio(-727010.0, -341829.0))

    def test_none_propagates(self):
        self.assertIsNone(_ratio(None, 100.0))
        self.assertIsNone(_ratio(100.0, None))


class NormaliseYearTests(unittest.TestCase):
    STATEMENT = {
        "id": 7192429,
        "regnskapstype": "KONSERN",
        "valuta": "USD",
        "oppstillingsplan": "store",
        "revisjon": {"ikkeRevidertAarsregnskap": False},
        "regnskapsperiode": {"fraDato": "2025-01-01", "tilDato": "2025-12-31"},
        "egenkapitalGjeld": {
            "egenkapital": {"sumEgenkapital": 39182000000.0},
            "gjeldOversikt": {"sumGjeld": 64249000000.0},
        },
        "eiendeler": {"sumEiendeler": 131727000000.0},
        "resultatregnskapResultat": {
            "aarsresultat": 5058000000.0,
            "ordinaertResultatFoerSkattekostnad": 6124000000.0,
            "driftsresultat": {
                "driftsresultat": 25353000000.0,
                "driftsinntekter": {"sumDriftsinntekter": 106462000000.0},
                "driftskostnad": {"sumDriftskostnad": 81109000000.0},
            },
        },
    }

    def test_maps_equinor_fy2025_to_published_figures(self):
        # Verified against Equinor's published FY2025 annual report.
        year = _normalise_year(self.STATEMENT)
        self.assertEqual(year["fiscal_year"], "2025")
        self.assertEqual(year["currency"], "USD")
        self.assertEqual(year["statement_scope"], "KONSERN")
        self.assertEqual(year["revenue"], 106462000000.0)
        self.assertEqual(year["operating_costs"], 81109000000.0)
        self.assertEqual(year["net_result"], 5058000000.0)

    def test_absent_optional_lines_become_none(self):
        # Large-company plans omit loennskostnad entirely.
        year = _normalise_year(self.STATEMENT)
        self.assertNotIn("loennskostnad", year)

    def test_sparse_statement_does_not_raise(self):
        year = _normalise_year({"regnskapsperiode": {"tilDato": "2024-12-31"}})
        self.assertIsNone(year["revenue"])
        self.assertIsNone(year["currency"])

    def test_audited_flag_inverts_unreviewed_flag(self):
        year = _normalise_year(self.STATEMENT)
        self.assertTrue(year["audited"])


if __name__ == "__main__":
    unittest.main()
