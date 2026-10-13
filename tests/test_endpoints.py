"""
Endpoint contracts.

These pin the exact URLs the agent depends on. A wrong path segment
returns 404 rather than raising, and because enrichment stages degrade to
empty results, a typo would otherwise surface as "this company has no
management and no accounts" for every company — a silent, total failure
that still looks like a successful run.
"""

import unittest
from unittest.mock import MagicMock, patch

import requests

from src.brreg import ENHETSREGISTERET_BASE, _entity_url
from src.financials import REGNSKAPSREGISTERET_BASE, _accounts_url
from src.roles import _roles_url

ORG = "989061593"

EXPECTED_ENTITY = (
    "https://data.brreg.no/enhetsregisteret/api/enheter/989061593"
)
EXPECTED_ROLES = (
    "https://data.brreg.no/enhetsregisteret/api/enheter/989061593/roller"
)
EXPECTED_ACCOUNTS = (
    "https://data.brreg.no/regnskapsregisteret/regnskap/989061593"
)


class UrlTests(unittest.TestCase):
    def test_entity_url(self):
        self.assertEqual(_entity_url(ORG), EXPECTED_ENTITY)

    def test_roles_url_uses_norwegian_path_segment(self):
        self.assertEqual(_roles_url(ORG), EXPECTED_ROLES)
        self.assertTrue(_roles_url(ORG).endswith("/roller"))

    def test_accounts_url(self):
        self.assertEqual(_accounts_url(ORG), EXPECTED_ACCOUNTS)

    def test_base_urls_are_the_official_hosts(self):
        self.assertEqual(
            ENHETSREGISTERET_BASE,
            "https://data.brreg.no/enhetsregisteret/api",
        )
        self.assertEqual(
            REGNSKAPSREGISTERET_BASE,
            "https://data.brreg.no/regnskapsregisteret/regnskap",
        )


class RolesRequestTests(unittest.TestCase):
    """A 404 must be reported, not silently rendered as 'no management'."""

    # The HTTP call lives in src.http; roles.py delegates to it. Patching
    # there keeps the test honest about where the request is issued.
    @patch("src.http.requests.get")
    def test_http_error_is_recorded_as_an_error(self, mock_get):
        mock_get.return_value.raise_for_status.side_effect = (
            requests.HTTPError("404 Not Found")
        )
        from src.roles import get_roles

        result = get_roles(ORG)
        self.assertEqual(result["roles"], [])
        self.assertIn("error", result)

    @patch("src.http.requests.get")
    def test_request_goes_to_the_roller_path(self, mock_get):
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"rollegrupper": []}
        mock_get.return_value = response

        from src.roles import get_roles

        get_roles(ORG)
        called_url = mock_get.call_args[0][0]
        self.assertEqual(called_url, EXPECTED_ROLES)


if __name__ == "__main__":
    unittest.main()