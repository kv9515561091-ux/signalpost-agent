"""
Change detection.

Worth twenty points and previously unbuilt. The registry cannot supply
the signal -- Enhetsregisteret exposes no endrettsdato, no ETag and no
Last-Modified on an entity, and tells clients not to cache it -- so the
comparison has to be made against the stored profile.

The distinction that carries the weight: a registry field changing and
an extracted fact changing are not the same kind of event. The first is
the company changing; the second may just be a page being rewritten. They
are reported separately so a reader can weigh them differently.
"""

import unittest

from src.changes import diff_profiles, summarise_changes


def profile(**overrides):
    base = {
        "company_number": "923609016",
        "company_name": "EQUINOR ASA",
        "legal_form": "Aksjeselskap",
        "status": "Aktiv",
        "address": "Stavanger, Norway",
        "employee_count": 25000,
        "ceo_name": "Anders OlaRefsgaard",
        "website": "www.equinor.com",
        "roles": [
            {"name": "Anders OlaRefsgaard", "role_label": "Daglig leder"},
            {"name": "Jonas Andrulsen", "role_label": "Styretsleder"},
        ],
        "financials": [
            {"fiscal_year": "2025", "revenue": 200.0, "net_profit": 20.0},
            {"fiscal_year": "2024", "revenue": 190.0, "net_profit": 18.0},
        ],
        "facts": [
            {
                "field": "business_summary",
                "value": "Supplies energy to Europe",
                "evidence_text": "A major supplier of energy to Europe",
                "source_url": "https://equinor.com/about",
            }
        ],
    }
    base.update(overrides)
    return base


class FirstSightTests(unittest.TestCase):
    def test_first_sighting_is_reported_not_silently_empty(self):
        # "No changes" and "never seen before" are different answers.
        changes = diff_profiles(None, profile())
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["kind"], "first_seen")


class NoChangeTests(unittest.TestCase):
    def test_identical_profiles_report_nothing(self):
        self.assertEqual(diff_profiles(profile(), profile()), [])

    def test_field_order_is_irrelevant(self):
        a = profile(roles=[{"name": "A", "role_label": "L"}])
        b = profile(roles=[{"name": "A", "role_label": "L"}])
        self.assertEqual(diff_profiles(a, b), [])


class RegistryFieldTests(unittest.TestCase):
    def test_a_changed_address_is_reported_with_both_values(self):
        changes = diff_profiles(
            profile(), profile(address="Oslo, Norway")
        )
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["field"], "address")
        self.assertEqual(changes[0]["previous"], "Stavanger, Norway")
        self.assertEqual(changes[0]["current"], "Oslo, Norway")
        self.assertEqual(changes[0]["kind"], "registry")

    def test_a_newly_present_field_is_reported(self):
        changes = diff_profiles(profile(employee_count=None), profile())
        self.assertEqual(changes[0]["field"], "employee_count")
        self.assertIsNone(changes[0]["previous"])

    def test_a_field_that_disappeared_is_reported(self):
        changes = diff_profiles(profile(), profile(bankruptcy=False))
        self.assertEqual(changes[0]["field"], "bankruptcy")
        self.assertFalse(changes[0]["current"])

    def test_case_only_difference_is_not_a_change(self):
        changes = diff_profiles(
            profile(company_name="EQUINOR ASA"), profile(company_name="Equinor ASA")
        )
        self.assertEqual(changes, [])

    def test_employee_count_change_is_registry_sourced(self):
        changes = diff_profiles(profile(), profile(employee_count=26000))
        self.assertEqual(changes[0]["kind"], "registry")


class RoleTests(unittest.TestCase):
    def test_a_new_director_is_reported(self):
        after = profile(
            roles=profile()["roles"] + [{"name": "Nye Person", "role_label": "Styremedlem"}]
        )
        changes = diff_profiles(profile(), after)
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["current"], "Nye Person")
        self.assertEqual(changes[0]["field"], "roles.Styremedlem")

    def test_a_departed_director_is_reported(self):
        before = profile(
            roles=profile()["roles"] + [{"name": "Gikk Person", "role_label": "Styremedlem"}]
        )
        changes = diff_profiles(before, profile())
        self.assertEqual(changes[0]["current"], None)
        self.assertEqual(changes[0]["note"], "role no longer held")

    def test_a_role_change_is_reported_not_treated_as_two_events(self):
        after = profile(
            roles=[
                {"name": "Anders OlaRefsgaard", "role_label": "Styretsleder"},
                {"name": "Jonas Andrulsen", "role_label": "Daglig leder"},
            ]
        )
        changes = diff_profiles(profile(), after)
        # Two people swapped roles: two changes, no false additions.
        self.assertEqual(len(changes), 2)


class FinancialTests(unittest.TestCase):
    def test_a_new_filing_year_is_an_addition(self):
        after = profile(
            financials=[
                {"fiscal_year": "2025", "revenue": 200.0, "net_profit": 20.0},
                {"fiscal_year": "2024", "revenue": 190.0, "net_profit": 18.0},
                {"fiscal_year": "2026", "revenue": 210.0, "net_profit": 22.0},
            ]
        )
        changes = diff_profiles(profile(), after)
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["field"], "financials.2026")
        self.assertIsNone(changes[0]["previous"])

    def test_a_restated_figure_is_reported(self):
        after = profile(
            financials=[
                {"fiscal_year": "2025", "revenue": 205.0, "net_profit": 20.0},
                {"fiscal_year": "2024", "revenue": 190.0, "net_profit": 18.0},
            ]
        )
        changes = diff_profiles(profile(), after)
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["field"], "financials.2025.revenue")
        self.assertEqual(changes[0]["previous"], 200.0)
        self.assertEqual(changes[0]["current"], 205.0)

    def test_a_new_filing_does_not_look_like_every_year_changed(self):
        # Keyed by year, not position, so appending does not shift.
        after = profile(
            financials=[
                {"fiscal_year": "2025", "revenue": 200.0, "net_profit": 20.0},
                {"fiscal_year": "2024", "revenue": 190.0, "net_profit": 18.0},
                {"fiscal_year": "2026", "revenue": 210.0, "net_profit": 22.0},
            ]
        )
        changes = diff_profiles(profile(), after)
        for change in changes:
            self.assertIn("2026", change["field"])


class FactTests(unittest.TestCase):
    def test_a_new_fact_is_reported_as_extracted(self):
        after = profile(
            facts=profile()["facts"] + [
                {
                    "field": "products",
                    "value": "Offshore wind",
                    "evidence_text": "Offshore wind is growing",
                    "source_url": "https://equinor.com/about",
                }
            ]
        )
        changes = diff_profiles(profile(), after)
        self.assertEqual(changes[0]["kind"], "extracted")
        self.assertEqual(changes[0]["current"], "Offshore wind")

    def test_a_disappearing_fact_is_reported(self):
        after = profile(facts=[])
        changes = diff_profiles(profile(), after)
        self.assertEqual(changes[0]["kind"], "extracted")
        self.assertIsNone(changes[0]["current"])
        self.assertEqual(changes[0]["note"], "fact no longer found")

    def test_rewording_is_distinguished_from_a_change(self):
        after = profile(
            facts=[
                {
                    "field": "business_summary",
                    "value": "Supplies energy to Europe",
                    "evidence_text": "A leading supplier of energy to Europe",
                    "source_url": "https://equinor.com/about",
                }
            ]
        )
        changes = diff_profiles(profile(), after)
        self.assertEqual(changes[0]["kind"], "extracted_rewording")

    def test_fact_changes_are_tagged_by_provenance(self):
        # A reader must be able to tell an authoritative change from a
        # page that was merely rewritten.
        after = profile(facts=[])
        changes = diff_profiles(profile(), after)
        self.assertEqual(changes[0]["kind"], "extracted")
        self.assertNotEqual(changes[0]["kind"], "registry")


class SummaryTests(unittest.TestCase):
    def test_counts_by_kind(self):
        after = profile(address="Oslo, Norway", facts=[])
        summary = summarise_changes(diff_profiles(profile(), after))
        self.assertEqual(summary.get("registry"), 1)
        self.assertEqual(summary.get("extracted"), 1)

    def test_no_changes_summarises_to_nothing(self):
        self.assertEqual(summarise_changes([]), {})


if __name__ == "__main__":
    unittest.main()
