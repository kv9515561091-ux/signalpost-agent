"""
End-to-end extraction through the OpenAI-compatible backend.

These exercise the path a user with only an NVIDIA key will take, with
the HTTP layer mocked. The point is to prove that switching provider
changes nothing about the safety guarantees: quoted evidence is still
verified against the page, and registry-owned fields are still
off-limits.
"""

import json
import unittest
from unittest.mock import MagicMock, patch

from support import disable_llm_pacing, restore_llm_pacing

PAGE = """
- G- GROUP AS
About us
We are a Norwegian investment company founded in 2005.
We invest in real estate and provide financial advisory services
to businesses across Norway.
Our head office is in Oslo.
Org.nr: 989061593
"""

ORG = "989061593"


def nvidia_response(payload, finish="stop"):
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "model": "nvidia/nemotron-3-super-120b-a12b",
        "choices": [
            {
                "finish_reason": finish,
                "message": {"content": json.dumps(payload)},
            }
        ],
        "usage": {"prompt_tokens": 1200, "completion_tokens": 200},
    }
    response.raise_for_status = lambda: None
    return response


def setUpModule():
    global _PACING
    _PACING = disable_llm_pacing()


def tearDownModule():
    restore_llm_pacing(_PACING)


class NvidiaExtractionTests(unittest.TestCase):
    def _extract(self, model_reply, provider="nvidia"):
        post = MagicMock(return_value=nvidia_response(model_reply))

        with patch.dict("os.environ", {"NVIDIA_API_KEY": "nvapi-test"}, clear=True):
            with patch("src.llm_openai.requests.post", post):
                from src.extract import extract_from_page

                return extract_from_page(
                    page_text=PAGE,
                    source_url="https://example.no/om-oss",
                    retrieved_at="2026-10-13T00:00:00+00:00",
                    company_number=ORG,
                    company_name="-G- GROUP AS",
                    tier=2,
                    source_type="official_company_website",
                    title="Om oss",
                )

    def test_verified_facts_survive_the_round_trip(self):
        report = self._extract(
            {
                "facts": [
                    {
                        "field": "business_summary",
                        "value": "Investment company in real estate",
                        "quote": "We are a Norwegian investment company founded in 2005.",
                        "published_date": "",
                    },
                    {
                        "field": "services",
                        "value": "Financial advisory",
                        "quote": "provide financial advisory services",
                        "published_date": "",
                    },
                ]
            }
        )
        self.assertTrue(report.llm_used)
        self.assertEqual(len(report.facts), 2)
        self.assertEqual(report.facts[0].field, "business_summary")
        self.assertEqual(report.facts[1].value, "Financial advisory")
        self.assertEqual(report.rejected, [])

    def test_hallucinated_quote_is_still_discarded(self):
        # The model claiming to have read the page must still be checked
        # against the page. A different provider does not relax this.
        report = self._extract(
            {
                "facts": [
                    {
                        "field": "markets_served",
                        "value": "12 countries",
                        "quote": "We operate in 12 countries across Europe.",
                        "published_date": "",
                    }
                ]
            }
        )
        self.assertEqual(report.facts, [])
        self.assertEqual(len(report.rejected), 1)
        self.assertEqual(
            report.rejected[0]["reason"], "quote not found in page"
        )

    def test_fenced_json_is_parsed_and_verified(self):
        block = (
            "Here are the facts:\n```json\n"
            + json.dumps(
                {
                    "facts": [
                        {
                            "field": "business_summary",
                            "value": "Investment company",
                            "quote": "We are a Norwegian investment company founded in 2005.",
                            "published_date": "",
                        }
                    ]
                }
            )
            + "\n```"
        )
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "choices": [
                {"finish_reason": "stop", "message": {"content": block}}
            ]
        }
        response.raise_for_status = lambda: None

        with patch.dict("os.environ", {"NVIDIA_API_KEY": "nvapi-test"}, clear=True):
            with patch("src.llm_openai.requests.post", MagicMock(return_value=response)):
                from src.extract import extract_from_page

                report = extract_from_page(
                    page_text=PAGE,
                    source_url="https://example.no/om-oss",
                    retrieved_at="t",
                    company_number=ORG,
                    company_name="-G- GROUP AS",
                    tier=2,
                    source_type="official_company_website",
                )
        self.assertEqual(len(report.facts), 1)

    def test_registry_owned_fields_are_refused(self):
        report = self._extract(
            {
                "facts": [
                    {
                        "field": "employee_count",
                        "value": "5000",
                        "quote": "We are a Norwegian investment company founded in 2005.",
                        "published_date": "",
                    }
                ]
            }
        )
        self.assertEqual(report.facts, [])
        self.assertIn("registry", report.rejected[0]["reason"])

    def test_transport_failure_reports_instead_of_raising(self):
        with patch.dict("os.environ", {"NVIDIA_API_KEY": "nvapi-test"}, clear=True):
            with patch(
                "src.llm_openai.requests.post",
                side_effect=Exception("network down"),
            ):
                with patch("src.llm_openai.time.sleep"):
                    from src.extract import extract_from_page

                    report = extract_from_page(
                        page_text=PAGE,
                        source_url="https://example.no/om-oss",
                        retrieved_at="t",
                        company_number=ORG,
                        company_name="-G- GROUP AS",
                        tier=2,
                        source_type="official_company_website",
                    )
        self.assertEqual(report.facts, [])
        self.assertIsNotNone(report.error)

    def test_confidence_is_still_capped_by_tier(self):
        report = self._extract(
            {
                "facts": [
                    {
                        "field": "business_summary",
                        "value": "Investment company",
                        "quote": "We are a Norwegian investment company founded in 2005.",
                        "published_date": "",
                    }
                ]
            }
        )
        self.assertEqual(report.facts[0].confidence, 0.90)


class NvidiaNarrativeTests(unittest.TestCase):
    def test_explanation_uses_the_backend_and_keeps_provenance(self):
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": "-G- GROUP AS invests in property in Oslo."
                    },
                }
            ]
        }
        response.raise_for_status = lambda: None

        from src.agent import CompanyProfile
        from src.explain import explain

        profile = CompanyProfile(
            company_number=ORG,
            company_name="-G- GROUP AS",
            legal_form="Aksjeselskap",
            sources=["https://data.brreg.no/x"],
        )

        with patch.dict("os.environ", {"NVIDIA_API_KEY": "nvapi-test"}, clear=True):
            with patch("src.llm_openai.requests.post", MagicMock(return_value=response)):
                result = explain(profile, use_llm=True)

        self.assertTrue(result.llm_used)
        # The attribution must name the provider that actually answered.
        # Hardcoding "claude" here misreported every non-Anthropic run.
        self.assertIn("openai_compatible", result.generated_by)
        self.assertIn("nvidia", result.generated_by)
        self.assertIn("openai_compatible", result.note)
        self.assertNotIn("Claude", result.note)
        self.assertTrue(
            result.text.startswith("-G- GROUP AS invests in property")
        )
        # The deterministic provenance section is still appended.
        self.assertIn("Brønnøysundregistrene", result.text)

    def test_narrative_citing_another_org_is_rejected(self):
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": "This is actually 983887457 doing it."},
                }
            ]
        }
        response.raise_for_status = lambda: None

        from src.agent import CompanyProfile
        from src.explain import explain

        profile = CompanyProfile(company_number=ORG, company_name="-G- GROUP AS")

        with patch.dict("os.environ", {"NVIDIA_API_KEY": "nvapi-test"}, clear=True):
            with patch("src.llm_openai.requests.post", MagicMock(return_value=response)):
                result = explain(profile, use_llm=True)

        self.assertFalse(result.llm_used)
        self.assertIn("another organization", result.note)


if __name__ == "__main__":
    unittest.main()
