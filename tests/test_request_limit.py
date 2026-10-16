"""
The request ceiling is configurable, and the two readers agree.

``AGENT_REQUEST_LIMIT`` used to be read only by ``src/budget.py`` while
the tracker that actually counts requests hardcoded 2,000. Setting the
variable for a larger build therefore appeared to work and changed
nothing -- a thousand-company run stopped enriching after two hundred
requests, with the run still reporting against a ceiling it had never
adopted.

A limit that is read in one place and ignored in another is worse than
no configuration at all, because the operator believes they changed
something.
"""

import os
import unittest
from unittest.mock import patch

from src.request_tracker import DEFAULT_LIMIT, RequestTracker, default_limit


class DefaultTests(unittest.TestCase):
    def test_competition_ceiling_is_the_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(default_limit(), 2000)
            self.assertEqual(DEFAULT_LIMIT, 2000)

    def test_env_overrides(self):
        with patch.dict(os.environ, {"AGENT_REQUEST_LIMIT": "25000"}):
            self.assertEqual(default_limit(), 25000)

    def test_garbage_falls_back_rather_than_crashing_a_run(self):
        with patch.dict(os.environ, {"AGENT_REQUEST_LIMIT": "many"}):
            self.assertEqual(default_limit(), DEFAULT_LIMIT)

    def test_zero_is_not_treated_as_unlimited(self):
        # Zero would disable the ceiling entirely, which is exactly the
        # sort of value a typo should not be able to produce.
        with patch.dict(os.environ, {"AGENT_REQUEST_LIMIT": "0"}):
            self.assertEqual(default_limit(), DEFAULT_LIMIT)

    def test_negative_is_not_treated_as_unlimited(self):
        with patch.dict(os.environ, {"AGENT_REQUEST_LIMIT": "-5"}):
            self.assertEqual(default_limit(), DEFAULT_LIMIT)

    def test_empty_is_the_default(self):
        with patch.dict(os.environ, {"AGENT_REQUEST_LIMIT": ""}):
            self.assertEqual(default_limit(), DEFAULT_LIMIT)


class TrackerTests(unittest.TestCase):
    def test_tracker_picks_up_the_environment(self):
        with patch.dict(os.environ, {"AGENT_REQUEST_LIMIT": "25000"}):
            self.assertEqual(RequestTracker().limit, 25000)

    def test_explicit_limit_beats_the_environment(self):
        with patch.dict(os.environ, {"AGENT_REQUEST_LIMIT": "25000"}):
            self.assertEqual(RequestTracker(limit=99).limit, 99)

    def test_budget_and_tracker_agree_on_the_ceiling(self):
        # The failure this exists for: two readers, one number.
        from src.budget import RequestBudget

        with patch.dict(os.environ, {"AGENT_REQUEST_LIMIT": "17000"}):
            self.assertEqual(
                RequestBudget().limit,
                RequestTracker().limit,
                "the tracker and the budget disagree about the ceiling",
            )

    def test_budget_follows_a_later_environment_change(self):
        from src.budget import RequestBudget

        with patch.dict(os.environ, {"AGENT_REQUEST_LIMIT": "17000"}):
            self.assertEqual(RequestBudget().limit, 17000)


if __name__ == "__main__":
    unittest.main()
