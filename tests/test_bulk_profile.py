"""Bulk identity records bypass live registry calls."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from src.agent import build_profile
from src.contract import envelope_for

COMPANY = {
    "company_number": "889640782",
    "company_name": "EKSEMPEL AS",
    "previous_names": [],
    "organization_form": "Aksjeselskap",
    "organization_form_code": "AS",
    "status": "aktiv",
    "industry": {"code": "62.010", "description": "Programmeringstjenester"},
    "sector": {"code": "2100", "description": "Private aksjeselskaper mv."},
    "address": "Testveien 1, 0150 OSLO, Norge",
    "municipality": "OSLO",
    "employee_count": 12,
    "website": "www.eksempel.no",
    "source": "https://data.brreg.no/enhetsregisteret/api/enheter/lastned/csv",
    "retrieved_at": "2026-11-03T00:00:00Z",
    "last_filed_annual_accounts": None,
}

ROLES = {
    "source": "https://data.brreg.no/enhetsregisteret/api/enheter/889640782/roller",
    "retrieved_at": "2026-11-03T00:00:00Z",
    "roles": [],
}


class BulkProfileTests(unittest.TestCase):
    def test_identity_and_missing_financials_skip_live_calls(self) -> None:
        with (
            patch("src.agent.get_company") as get_company,
            patch("src.agent.get_roles", return_value=ROLES),
            patch("src.agent.get_financials") as get_financials,
        ):
            profile = build_profile(
                "889640782",
                company_record=COMPANY,
                include_financials=False,
                include_narrative=False,
                use_llm=False,
            )

        get_company.assert_not_called()
        get_financials.assert_not_called()
        self.assertEqual(profile.company_name, "EKSEMPEL AS")
        self.assertEqual(profile.financials, [])

    def test_registered_subunits_become_workplaces_and_claims(self) -> None:
        subunits = [
            {
                "organisation_number": "912345678",
                "name": "EKSEMPEL AVDELING",
                "address": "Avdelingsveien 2, 5003 BERGEN, Norge",
                "municipality": "BERGEN",
                "employee_count": 8,
                "source": "https://data.brreg.no/enhetsregisteret/api/underenheter/lastned/csv",
                "retrieved_at": "2026-11-03T00:00:00Z",
            }
        ]
        profile = build_profile(
            "889640782",
            company_record=COMPANY,
            subunits=subunits,
            include_roles=False,
            include_financials=False,
            include_narrative=False,
            use_llm=False,
        )

        self.assertEqual(profile.locations[0]["name"], "EKSEMPEL AVDELING")
        fields = {item.field for item in profile.evidence}
        self.assertIn("workplace_name", fields)
        self.assertIn("workplace_address", fields)
        self.assertIn("workplace_employee_count", fields)

        envelope = envelope_for(
            profile.model_dump(mode="json"),
            run_id="test-run",
            started_at="2026-11-03T00:00:00Z",
            completed_at="2026-11-03T00:01:00Z",
        )
        self.assertEqual(envelope["modules"]["locations"]["state"], "complete")
        claim_fields = {claim["field"] for claim in envelope["claims"]}
        self.assertIn("workplace_name", claim_fields)
        source_classes = {
            item["source_class"] for item in envelope["evidence"]
        }
        self.assertIn("government_registry", source_classes)
