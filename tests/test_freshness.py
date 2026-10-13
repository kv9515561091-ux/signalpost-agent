"""Freshness classification semantics."""

import unittest
from datetime import date, timedelta

from src.freshness import (
    STATUS_AGING,
    STATUS_CURRENT,
    STATUS_STALE,
    STATUS_UNKNOWN,
    age_in_days,
    assess,
)


def days_ago(n: int) -> str:
    return (date.today() - timedelta(days=n)).isoformat()


class AgeTests(unittest.TestCase):
    def test_computes_age_from_iso_date(self):
        self.assertAlmostEqual(age_in_days(days_ago(10)), 10.0, delta=1)

    def test_understands_norwegian_date_format(self):
        self.assertIsNotNone(age_in_days("23.03.2006"))

    def test_unknown_dates_return_none(self):
        self.assertIsNone(age_in_days(None))
        self.assertIsNone(age_in_days(""))


class AssessTests(unittest.TestCase):
    def test_live_registry_is_current_by_definition(self):
        # A continuously maintained record read now cannot be stale, even
        # though it publishes no per-field update timestamp.
        result = assess(None, "live_registry")
        self.assertEqual(result["status"], STATUS_CURRENT)
        self.assertIsNone(result["window_days"])

    def test_financial_period_within_window_is_current(self):
        result = assess(days_ago(30), "financial")
        self.assertEqual(result["status"], STATUS_CURRENT)

    def test_financial_period_past_double_window_is_stale(self):
        result = assess(days_ago(1500), "financial")
        self.assertEqual(result["status"], STATUS_STALE)

    def test_role_unchanged_for_years_is_stale(self):
        result = assess("2006-01-23", "registry")
        self.assertEqual(result["status"], STATUS_STALE)

    def test_middle_band_is_aging(self):
        result = assess(days_ago(180), "registry")
        self.assertEqual(result["status"], STATUS_AGING)

    def test_missing_date_for_periodic_fact_is_unknown(self):
        result = assess(None, "financial")
        self.assertEqual(result["status"], STATUS_UNKNOWN)

    def test_unknown_kind_defaults_to_registry_window(self):
        result = assess(days_ago(5), "not-a-kind")
        self.assertEqual(result["window_days"], 120)


if __name__ == "__main__":
    unittest.main()
