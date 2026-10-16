"""
Request budget.

Exceeding the daily limit disqualifies the entry, so the agent must
notice it is running out rather than discovering it as an exception
mid-company. Two failure modes bracket the fix: dying on the limit throws
away finished work, and never noticing until the exception propagates
takes every remaining company down with it.

The registry is the last thing cut. It is free, three requests per
company, and every profile depends on it, so a budget drained to its
floor still produces a valid profile for every company rather than an
exception.
"""

import unittest

from src.budget import DEFAULT_LIMIT, REGISTRY_REQUESTS_PER_COMPANY, RequestBudget


class ConfigurationTests(unittest.TestCase):
    def test_default_matches_the_stated_daily_ceiling(self):
        self.assertEqual(DEFAULT_LIMIT, 2000)

    def test_registry_cost_is_three_per_company(self):
        self.assertEqual(REGISTRY_REQUESTS_PER_COMPANY, 3)


class SpendTests(unittest.TestCase):
    def test_fresh_budget_is_untouched(self):
        budget = RequestBudget(100)
        self.assertEqual(budget.used, 0)
        self.assertEqual(budget.remaining, 100)
        self.assertFalse(budget.exhausted())

    def test_spending_reduces_remaining(self):
        budget = RequestBudget(100)
        budget.spend(30)
        self.assertEqual(budget.remaining, 70)

    def test_remaining_never_goes_negative(self):
        budget = RequestBudget(10)
        budget.spend(50)
        self.assertEqual(budget.remaining, 0)

    def test_zero_limit_is_exhausted(self):
        self.assertTrue(RequestBudget(0).exhausted())

    def test_fraction_used(self):
        budget = RequestBudget(200)
        budget.spend(50)
        self.assertAlmostEqual(budget.fraction_used, 0.25)


class DegradationTests(unittest.TestCase):
    def test_full_budget_grants_full_depth(self):
        budget = RequestBudget(2000)
        self.assertEqual(budget.pages_allowed(16, 17, 100), 16)

    def test_tight_budget_reduces_depth(self):
        # 500 left, 100 companies: 5 each against a desired 17.
        budget = RequestBudget(500)
        self.assertEqual(budget.pages_allowed(17, 17, 100), 5)

    def test_depth_is_never_less_than_one_when_it_can_be_afforded(self):
        # One page is the minimum useful amount, and the floor only
        # applies while a single company can be paid for.
        budget = RequestBudget(200)
        budget.used = 199
        self.assertGreaterEqual(budget.pages_allowed(17, 17, 1), 1)

    def test_no_pages_when_budget_cannot_reach_every_company(self):
        # 1 request left and 100 companies still to do: giving one page
        # each would overspend, so enrichment stops for the tail rather
        # than starving the first company and leaving the rest blind.
        budget = RequestBudget(200)
        budget.used = 199
        self.assertEqual(budget.pages_allowed(17, 17, 100), 0)

    def test_no_budget_grants_no_pages(self):
        budget = RequestBudget(200)
        budget.used = 200
        self.assertEqual(budget.pages_allowed(17, 17, 100), 0)

    def test_degradation_is_monotonic(self):
        """Depth must never increase as the budget drains."""
        granted = []
        for used in range(0, 1000, 50):
            budget = RequestBudget(1000)
            budget.used = used
            granted.append(budget.pages_allowed(16, 17, 50))
        self.assertEqual(granted, sorted(granted, reverse=True))

    def test_enrichment_survives_on_the_registry_reserve(self):
        """The floor is what the registry needs for every company."""
        budget = RequestBudget(2000, protected=REGISTRY_REQUESTS_PER_COMPANY * 100)

        # 400 left against a 300 reserve: enrichment can still afford
        # to run, though barely.
        budget.used = 1600
        self.assertFalse(budget.registry_only())

        # 250 left: the reserve alone would not cover the registry, so
        # enrichment stops rather than starving the spine.
        budget.used = 1750
        self.assertTrue(budget.registry_only())


class SmoothDivisionTests(unittest.TestCase):
    def test_remainder_is_shared_evenly(self):
        budget = RequestBudget(2000)
        budget.used = 1700  # 300 left
        self.assertEqual(budget.reserve(17, 100), 3)

    def test_no_companies_left_divides_by_nothing(self):
        budget = RequestBudget(2000)
        self.assertEqual(budget.reserve(17, 0), 0)

    def test_grants_never_exceed_the_budget(self):
        """The bug this class exists for.

        Without debiting as pages are promised, ten companies sharing
        five requests were granted 0,0,0,0,0,1,1,1,2,5 -- ten in total
        against five available. The remainder was re-promised to every
        remaining company because nothing recorded it as spoken for.
        """
        budget = RequestBudget(100)
        budget.used = 95  # 5 left

        granted_total = 0
        companies_left = 10
        while companies_left > 0:
            pages = budget.pages_allowed(17, 17, companies_left)
            granted_total += pages
            budget.commit(pages)
            companies_left -= 1

        self.assertLessEqual(granted_total, 5)

    def test_reservations_survive_a_sync(self):
        # The batch re-reads the tracker each company; that must not wipe
        # promises already made.
        from src.request_tracker import RequestTracker

        tracker = RequestTracker(100)
        tracker.count = 50

        budget = RequestBudget.from_tracker(tracker)
        budget.commit(3)
        budget.sync(tracker)

        self.assertEqual(budget.reserved, 3)
        self.assertEqual(budget.remaining, 47)

    def test_even_distribution_beats_concentration(self):
        """Coverage is scored per company, so spreading wins."""
        budget = RequestBudget(100)
        budget.used = 95
        enriched = 0
        companies_left = 10
        while companies_left > 0:
            if budget.pages_allowed(17, 17, companies_left) > 0:
                enriched += 1
                budget.commit(1)
            companies_left -= 1
        self.assertEqual(enriched, 5)


class TrackerAdoptionTests(unittest.TestCase):
    def test_adopts_the_tracker_limit_and_spend(self):
        from src.request_tracker import RequestTracker

        tracker = RequestTracker(1500)
        tracker.count = 400
        budget = RequestBudget.from_tracker(tracker, protected=300)

        self.assertEqual(budget.limit, 1500)
        self.assertEqual(budget.used, 400)
        self.assertEqual(budget.remaining, 1100)


class SummaryTests(unittest.TestCase):
    def test_summary_reports_the_ceiling_and_the_floor(self):
        budget = RequestBudget(2000, protected=300)
        budget.spend(800)
        summary = budget.summary()

        self.assertEqual(summary["budget_limit"], 2000)
        self.assertEqual(summary["budget_used"], 800)
        self.assertEqual(summary["budget_remaining"], 1200)
        self.assertEqual(summary["budget_used_pct"], 40.0)
        self.assertFalse(summary["enrichment_disabled_for_budget"])

    def test_summary_flags_when_only_the_registry_reserve_is_left(self):
        budget = RequestBudget(2000, protected=300)
        budget.spend(1750)
        self.assertTrue(budget.summary()["enrichment_disabled_for_budget"])


if __name__ == "__main__":
    unittest.main()
