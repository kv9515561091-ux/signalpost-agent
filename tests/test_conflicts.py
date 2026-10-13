"""Conflict detection and reconciliation."""

import unittest

from src.conflicts import FieldValue, conflict_report, reconcile, reconcile_all


def claim(value, tier=4, url="https://a.example", field="revenue", confidence=0.7):
    return FieldValue(
        field=field,
        value=value,
        source_url=url,
        retrieved_at="2026-10-13T00:00:00+00:00",
        tier=tier,
        confidence=confidence,
        source_type=f"tier{tier}",
    )


class AgreementTests(unittest.TestCase):
    def test_identical_values_are_not_a_conflict(self):
        result = reconcile("revenue", [claim(100), claim(100, url="https://b")])
        self.assertFalse(result.has_conflict)
        self.assertEqual(result.value, 100)

    def test_differently_formatted_equal_values_agree(self):
        result = reconcile(
            "ceo",
            [
                claim("Geir Inge Gjørva", field="ceo"),
                claim("geir inge gjorva", field="ceo", url="https://b"),
            ],
        )
        self.assertFalse(result.has_conflict)

    def test_single_claim_resolves_trivially(self):
        result = reconcile("revenue", [claim(100)])
        self.assertEqual(result.value, 100)
        self.assertFalse(result.has_conflict)

    def test_no_claims_yields_none(self):
        result = reconcile("revenue", [])
        self.assertIsNone(result.value)
        self.assertFalse(result.has_conflict)


class RegistryPrecedenceTests(unittest.TestCase):
    def test_registry_outranks_any_web_source(self):
        result = reconcile(
            "employee_count",
            [
                claim(50, tier=6, url="https://purehelp.no"),
                claim(1932, tier=1, url="https://data.brreg.no"),
                claim(60, tier=4, url="https://proff.no"),
            ],
        )
        self.assertEqual(result.value, 1932)
        self.assertTrue(result.has_conflict)
        self.assertEqual(
            result.conflict.resolution, "resolved_by_official_registry"
        )

    def test_conflict_records_every_dissenting_value(self):
        result = reconcile(
            "employee_count",
            [claim(50, tier=6), claim(1932, tier=1)],
        )
        values = {v.value for v in result.conflict.values}
        self.assertEqual(values, {50, 1932})


class NonRegistryDisagreementTests(unittest.TestCase):
    def test_higher_tier_wins_when_clearly_more_authoritative(self):
        result = reconcile(
            "phone",
            [claim("22 24 90 90", tier=4, url="https://proff.no"),
             claim("911 76 537", tier=6, url="https://purehelp.no")],
        )
        self.assertEqual(result.value, "22 24 90 90")
        self.assertEqual(result.conflict.resolution, "resolved_by_source_tier")

    def test_equal_authority_is_reported_not_decided(self):
        result = reconcile(
            "phone",
            [claim("22 24 90 90", tier=6, url="https://a.no"),
             claim("911 76 537", tier=6, url="https://b.no")],
        )
        self.assertTrue(result.has_conflict)
        self.assertEqual(
            result.conflict.resolution, "unresolved_sources_disagree"
        )
        self.assertEqual(len(result.conflict.values), 2)


class ReportingTests(unittest.TestCase):
    def test_report_omits_agreeing_fields(self):
        report = conflict_report(
            reconcile_all({"revenue": [claim(100), claim(100)]})
        )
        self.assertEqual(report, [])

    def test_report_includes_provenance_for_each_value(self):
        report = conflict_report(
            reconcile_all({"revenue": [claim(100, tier=1), claim(200, tier=6)]})
        )
        self.assertEqual(len(report), 1)
        entry = report[0]
        self.assertEqual(entry["field"], "revenue")
        self.assertEqual(entry["published_value"], 100)
        self.assertEqual(len(entry["values"]), 2)
        for value in entry["values"]:
            self.assertIn("source_url", value)
            self.assertIn("tier", value)


if __name__ == "__main__":
    unittest.main()
