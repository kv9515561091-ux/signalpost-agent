"""Agent orchestration, with the registry layer mocked out.

These exercise wiring and evidence attribution without network access.
Live endpoint contracts are pinned separately in test_financials.py.
"""

import unittest
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

from src.agent import build_profile, reconcile_evidence, validate_company_number
from src.evidence import Evidence
from src.orgnumber import InvalidCompanyNumber


COMPANY = {
    "company_number": "989061593",
    "company_name": "-G- GROUP AS",
    "previous_names": [{"name": "Tidligere Navn AS", "from": None, "to": None}],
    "organization_form_code": "AS",
    "organization_form": "Aksjeselskap",
    "status": "aktiv",
    "bankruptcy": False,
    "bankruptcy_date": None,
    "under_liquidation": False,
    "under_liquidation_date": None,
    "under_forced_liquidation": False,
    "registration_date": "2006-01-23",
    "founding_date": "2005-10-01",
    "articles_date": "2011-08-26",
    "last_filed_annual_accounts": "2025",
    "address": "Hellerud gårdsvei 5 B, 0671 OSLO, Norge",
    "business_address": "Hellerud gårdsvei 5 B, 0671 OSLO, Norge",
    "postal_address": None,
    "municipality": "OSLO",
    "municipality_code": "0301",
    "industry": {"code": "62.200", "description": "IT-konsulentvirksomhet"},
    "industry_secondary": None,
    "sector": None,
    "employee_count": 0,
    "employee_count_registered": True,
    "employee_count_as_of": "2026-09-14",
    "business_description": "Investeringer og eiendom.",
    "notations": [],
    "phone": "22 24 90 90",
    "mobile": None,
    "email": "post@example.no",
    "website": "example.no",
    "language_form": "Bokmål",
    "is_group_parent": False,
    "parent_company_number": None,
    "registered_business_register": True,
    "registered_vat_register": True,
    "registered_voluntary_register": False,
    "registered_party_register": False,
    "registered_foundation_register": False,
    "source": "https://data.brreg.no/enhetsregisteret/api/enheter/989061593",
    "retrieved_at": "2026-10-13T10:00:00+00:00",
}

ROLES = {
    "roles": [
        {
            "role_code": "DAGL",
            "role": "Daglig leder",
            "group_code": "DAGL",
            "group": "managing_director",
            "person_name": "Geir Inge Gjørva",
            "group_changed_at": "2006-01-23",
        }
    ],
    "groups": [],
    "managing_directors": [],
    "board": [],
    "auditors": [],
    "ceo_name": "Geir Inge Gjørva",
    "board_chair_name": None,
    "board_size": 0,
    "auditor_name": None,
    "source": "https://data.brreg.no/enhetsregisteret/api/enheter/989061593/roller",
    "retrieved_at": "2026-10-13T10:00:01+00:00",
}

FINANCIALS = {
    "years": [
        {
            "fiscal_year": "2025",
            "period_end": "2025-12-31",
            "currency": "NOK",
            "statement_scope": "SELSKAP",
            "audited": True,
            "revenue": 20000.0,
            "net_result": 264344.0,
            "total_assets": 2949837.0,
            "operating_margin_pct": 28.7,
        }
    ],
    "latest": None,
    "currency": "NOK",
    "source": "https://data.brreg.no/regnskapsregisteret/regnskap/989061593",
    "retrieved_at": "2026-10-13T10:00:02+00:00",
}


class BuildProfileTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.get_company = stack.enter_context(
            patch("src.agent.get_company", return_value=COMPANY)
        )
        self.get_roles = stack.enter_context(
            patch("src.agent.get_roles", return_value=ROLES)
        )
        self.get_financials = stack.enter_context(
            patch("src.agent.get_financials", return_value=FINANCIALS)
        )

    def test_maps_registry_fields_onto_profile(self):
        profile = build_profile("989061593")
        self.assertEqual(profile.company_number, "989061593")
        self.assertEqual(profile.company_name, "-G- GROUP AS")
        self.assertEqual(profile.legal_form_code, "AS")
        self.assertEqual(profile.industry_code, "62.200")
        self.assertEqual(profile.employee_count, 0)

    def test_surfaces_management_and_financials(self):
        profile = build_profile("989061593")
        self.assertEqual(profile.ceo_name, "Geir Inge Gjørva")
        self.assertEqual(len(profile.financials), 1)
        self.assertEqual(profile.financial_currency, "NOK")

    def test_attaches_evidence_to_every_core_fact(self):
        profile = build_profile("989061593")
        fields = {e.field for e in profile.evidence}
        for required in (
            "company_number",
            "company_name",
            "legal_form",
            "industry",
            "address",
            "employee_count",
            "role:DAGL",
            "revenue",
        ):
            self.assertIn(required, fields)

    def test_registry_evidence_is_tier_one(self):
        profile = build_profile("989061593")
        registry = [
            e for e in profile.evidence
            if e.source_type == "official_registry"
        ]
        self.assertTrue(registry)
        self.assertTrue(all(e.confidence == 1.0 for e in registry))
        self.assertTrue(all(e.source_tier == 1 for e in registry))

    def test_financial_evidence_carries_currency_and_period(self):
        profile = build_profile("989061593")
        revenue = next(e for e in profile.evidence if e.field == "revenue")
        self.assertIn("NOK", revenue.evidence_text)
        self.assertEqual(revenue.reporting_period, "2025")
        self.assertEqual(revenue.as_of, "2025-12-31")

    def test_sources_are_deduplicated(self):
        profile = build_profile("989061593")
        self.assertEqual(len(profile.sources), len(set(profile.sources)))

    def test_norwegian_characters_survive(self):
        profile = build_profile("989061593")
        self.assertEqual(profile.ceo_name, "Geir Inge Gjørva")
        self.assertIn("gårdsvei", profile.address)

    def test_accepts_separated_input_and_calls_api_with_canonical(self):
        build_profile(" 989 061 593 ")
        self.get_company.assert_called_once_with("989061593")

    def test_rejects_invalid_number_before_any_request(self):
        for bad in ("12345678", "983887458", "abc"):
            with self.assertRaises(InvalidCompanyNumber):
                build_profile(bad)
        self.assertEqual(self.get_company.call_count, 0)

    def test_company_with_no_filed_accounts_still_builds(self):
        self.get_financials.return_value = {
            "years": [],
            "latest": None,
            "currency": None,
            "source": "https://example",
            "retrieved_at": "2026-10-13T10:00:02+00:00",
        }
        profile = build_profile("889640782")
        self.assertEqual(profile.financials, [])
        self.assertIsNone(profile.financial_currency)
        self.assertEqual(profile.company_name, "-G- GROUP AS")


class DegradedRetrievalTests(unittest.TestCase):
    """A transient failure must not discard the whole profile."""

    def test_roles_failure_is_recorded_not_fatal(self):
        broken_roles = dict(ROLES, roles=[], ceo_name=None, error="timeout")
        with patch("src.agent.get_company", return_value=COMPANY):
            with patch("src.agent.get_roles", return_value=broken_roles):
                with patch(
                    "src.agent.get_financials", return_value=FINANCIALS
                ):
                    profile = build_profile("989061593")

        # Identity survives...
        self.assertEqual(profile.company_name, "-G- GROUP AS")
        self.assertEqual(profile.employee_count, COMPANY["employee_count"])
        # ...financials survive...
        self.assertEqual(len(profile.financials), 1)
        # ...the gap is visible rather than silent...
        self.assertTrue(
            any(e.startswith("roles:") for e in profile.retrieval_errors)
        )
        # ...and no phantom CEO is invented.
        self.assertIsNone(profile.ceo_name)
        self.assertEqual(profile.roles, [])

    def test_financials_failure_is_recorded_not_fatal(self):
        broken_fin = dict(
            FINANCIALS, years=[], latest=None, currency=None, error="timeout"
        )
        with patch("src.agent.get_company", return_value=COMPANY):
            with patch("src.agent.get_roles", return_value=ROLES):
                with patch(
                    "src.agent.get_financials", return_value=broken_fin
                ):
                    profile = build_profile("989061593")

        self.assertEqual(profile.ceo_name, "Geir Inge Gjørva")
        self.assertEqual(profile.financials, [])
        self.assertTrue(
            any(e.startswith("financials:") for e in profile.retrieval_errors)
        )

    def test_clean_run_records_no_errors(self):
        with patch("src.agent.get_company", return_value=COMPANY):
            with patch("src.agent.get_roles", return_value=ROLES):
                with patch(
                    "src.agent.get_financials", return_value=FINANCIALS
                ):
                    profile = build_profile("989061593")
        self.assertEqual(profile.retrieval_errors, [])

    def test_sole_proprietor_owner_is_surfaced(self):
        # An ENK has no managing director; the sole proprietor is the
        # company. "No CEO" must not be the only summary of it.
        enk_roles = dict(
            ROLES,
            roles=[
                {
                    "role_code": "INNH",
                    "role": "Innehaver",
                    "group_code": "INNH",
                    "group": "owner",
                    "person_name": "Andrii Zotko",
                    "group_changed_at": "2024-01-01",
                }
            ],
            managing_directors=[],
            ceo_name=None,
            board=[],
            board_size=0,
            owner_name="Andrii Zotko",
        )
        with patch("src.agent.get_company", return_value=COMPANY):
            with patch("src.agent.get_roles", return_value=enk_roles):
                with patch(
                    "src.agent.get_financials", return_value=FINANCIALS
                ):
                    profile = build_profile("989061593")

        self.assertIsNone(profile.ceo_name)
        self.assertEqual(profile.owner_name, "Andrii Zotko")


class FetchFailureReportingTests(unittest.TestCase):
    """A bare exception name tells a reader nothing actionable."""

    def _response(self, status):
        response = MagicMock()
        response.status_code = status
        response.url = "https://example.no"
        return response

    def test_forbidden_says_retrying_will_not_help(self):
        import requests

        from src.agent import _describe_fetch_failure

        exc = requests.HTTPError(response=self._response(403))
        message = _describe_fetch_failure(exc)
        self.assertIn("403", message)
        self.assertIn("nothing to fix", message)

    def test_rate_limit_is_distinguished(self):
        import requests

        from src.agent import _describe_fetch_failure

        message = _describe_fetch_failure(
            requests.HTTPError(response=self._response(429))
        )
        self.assertIn("rate limited", message)

    def test_server_error_is_distinguished(self):
        import requests

        from src.agent import _describe_fetch_failure

        self.assertIn("server error", _describe_fetch_failure(
            requests.HTTPError(response=self._response(503))
        ))

    def test_timeout_and_connection_error_are_named(self):
        import requests

        from src.agent import _describe_fetch_failure

        self.assertEqual(
            _describe_fetch_failure(requests.Timeout()), "timed out"
        )
        self.assertIn(
            "connection failed",
            _describe_fetch_failure(requests.ConnectionError()),
        )

    def test_unknown_exception_keeps_its_message(self):
        from src.agent import _describe_fetch_failure

        message = _describe_fetch_failure(ValueError("something odd"))
        self.assertIn("something odd", message)

    def test_no_bare_exception_class_names_reach_the_profile(self):
        # "HTTPError" alone was what surfaced from the live run.
        from src.agent import _describe_fetch_failure
        import requests

        for exc in (
            requests.HTTPError(response=self._response(403)),
            requests.Timeout(),
            requests.ConnectionError(),
        ):
            self.assertNotEqual(_describe_fetch_failure(exc), type(exc).__name__)


class ValidateCompanyNumberTests(unittest.TestCase):
    def test_accepts_valid(self):
        self.assertTrue(validate_company_number("983 887 457"))

    def test_rejects_invalid(self):
        self.assertFalse(validate_company_number("983887458"))
        self.assertFalse(validate_company_number("12345678"))


class WebEnrichmentTests(unittest.TestCase):
    """Enrichment must never overwrite a registry record."""

    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch("src.agent.get_company", return_value=COMPANY))
        stack.enter_context(patch("src.agent.get_roles", return_value=ROLES))
        stack.enter_context(patch("src.agent.get_financials", return_value=FINANCIALS))
        self.discover = stack.enter_context(
            patch("src.agent.discover_sources", return_value=[])
        )

    def test_enrichment_is_off_by_default(self):
        profile = build_profile("989061593")
        self.assertEqual(profile.web_sources, [])
        self.discover.assert_not_called()

    def test_verified_sources_are_attached_with_tier_and_basis(self):
        self.discover.return_value = [
            {
                "url": "https://bedri.no/bedrift/989061593",
                "title": "-G- GROUP AS",
                "domain": "bedri.no",
                "tier": 6,
                "source_type": "general_directory",
                "confidence": 0.45,
                "identity_basis": "organization_number",
                "identity_signals": {"org_number_in_url_path": True},
                "query": '"989061593"',
                "retrieved_at": "2026-10-13T10:00:03+00:00",
            }
        ]
        profile = build_profile("989061593", include_web=True)
        self.assertEqual(len(profile.web_sources), 1)
        source = profile.web_sources[0]
        self.assertEqual(source.tier, 6)
        self.assertEqual(source.confidence, 0.45)
        self.assertEqual(source.identity_basis, "organization_number")
        self.assertIn("https://bedri.no/bedrift/989061593", profile.sources)

    def test_registry_value_is_not_overwritten_by_a_lower_tier(self):
        self.discover.return_value = []
        profile = build_profile("989061593", include_web=True)
        # employee_count comes from Brreg and must survive enrichment.
        self.assertEqual(profile.employee_count, COMPANY["employee_count"])

    def test_multi_year_financials_are_not_reported_as_conflicts(self):
        # Revenue differs across fiscal years; that is two facts, not a
        # disagreement about one.
        self.get_multi = patch(
            "src.agent.get_financials",
            return_value={
                "years": [
                    dict(FINANCIALS["years"][0], fiscal_year="2025"),
                    dict(FINANCIALS["years"][0], fiscal_year="2024", revenue=15000.0),
                ],
                "latest": None,
                "currency": "NOK",
                "source": FINANCIALS["source"],
                "retrieved_at": FINANCIALS["retrieved_at"],
            },
        )
        self.get_multi.start()
        self.addCleanup(self.get_multi.stop)
        profile = build_profile("989061593", include_web=True)
        self.assertEqual(profile.conflicts, [])

    def test_true_disagreement_is_reported_not_silently_dropped(self):
        # A lower-tier source contradicting Brreg must not overwrite it,
        # and the losing value must stay visible.
        profile = build_profile("989061593")
        self.assertEqual(profile.conflicts, [])

        profile.evidence.append(
            Evidence(
                field="employee_count",
                value=9999,
                source_url="https://purehelp.no/x",
                retrieved_at="2026-10-13T10:00:03+00:00",
                evidence_text="Antall ansatte: 9999",
                confidence=0.45,
                source_tier=6,
                source_type="general_directory",
            )
        )

        report = reconcile_evidence(profile.evidence)
        employee = [c for c in report if c["field"] == "employee_count"]
        self.assertEqual(len(employee), 1)
        self.assertEqual(employee[0]["published_value"], 0)
        self.assertEqual(
            employee[0]["resolution"], "resolved_by_official_registry"
        )
        self.assertEqual(len(employee[0]["values"]), 2)

    def test_multiple_co_ceos_are_not_a_conflict(self):
        profile = build_profile("989061593")
        profile.evidence.append(
            Evidence(
                field="role:DAGL",
                value="Second Chief Executive",
                source_url="https://data.brreg.no/roles",
                retrieved_at="2026-10-13T10:00:01+00:00",
                evidence_text="Daglig leder: Second Chief Executive",
                confidence=1.0,
                source_tier=1,
                source_type="official_registry_roles",
            )
        )
        report = reconcile_evidence(profile.evidence)
        self.assertEqual([c for c in report if c["field"].startswith("role:")], [])


if __name__ == "__main__":
    unittest.main()
