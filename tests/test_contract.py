"""Contract adapter and runner invariants."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import contract, contract_cli


def _profile() -> dict:
    return {
        "company_number": "123456789",
        "company_name": "EXAMPLE AS",
        "roles": [{"role": "Daglig leder", "person_name": "Ada"}],
        "financials": [{"fiscal_year": "2025", "revenue": 1000.0}],
        "website": "https://example.no",
        "website_verified": True,
        "evidence": [
            {
                "field": "company_name",
                "value": "EXAMPLE AS",
                "source_url": "https://data.brreg.no/enheter/123456789",
                "retrieved_at": "2026-01-01T00:00:00Z",
                "evidence_text": "Registered legal name: EXAMPLE AS",
                "confidence": 1.0,
                "source_tier": 1,
                "source_type": "official_registry",
            }
        ],
        "facts": [
            {
                "field": "employee_count",
                "value": 12,
                "source_url": "https://example.no/about",
                "retrieved_at": "2026-01-01T00:00:01Z",
                "evidence_text": "Vi er 12 ansatte.",
                "confidence": 0.9,
                "source_tier": 2,
                "source_type": "official_company_website",
            }
        ],
        "changes": [{"kind": "first_seen", "field": "company_name"}],
        "retrieval_errors": ["financials: no filing"],
        "retrieved_at": "2026-01-01T00:00:02Z",
        "_operations": {"requests": 3, "runtime_ms": 1200},
    }


class InputTests(unittest.TestCase):
    def test_jsonl_objects_preserve_split_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "organisation_number": "123 456 789",
                        "evaluation_split": "development",
                        "sample_slice": "population",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            records = contract.read_inputs(path)
        self.assertEqual(records[0]["organisation_number"], "123456789")
        self.assertEqual(records[0]["evaluation_split"], "development")
        self.assertEqual(records[0]["sample_slice"], "population")

    def test_text_input_is_one_number_per_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.txt"
            path.write_text("\ufeff123456789\n\n987654321\n", encoding="utf-8")
            records = contract.read_inputs(path)
        self.assertEqual(
            [record["organisation_number"] for record in records],
            ["123456789", "987654321"],
        )

    def test_duplicates_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.txt"
            path.write_text("123456789\n123456789\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicates"):
                contract.read_inputs(path)

    def test_invalid_number_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.txt"
            path.write_text("123\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid"):
                contract.read_inputs(path)


class EnvelopeTests(unittest.TestCase):
    def test_envelope_has_both_contract_surfaces(self):
        envelope = contract.envelope_for(
            _profile(),
            run_id="run-1",
            started_at="2026-01-01T00:00:00Z",
            completed_at="2026-01-01T00:00:03Z",
        )
        self.assertEqual(envelope["organisation_number"], "123456789")
        self.assertEqual(envelope["run"]["terminal_status"], "completed")
        self.assertEqual(envelope["state"], "complete")
        self.assertEqual(envelope["run_id"], "run-1")
        self.assertEqual(len(envelope["claims"]), 2)
        self.assertEqual(len(envelope["evidence"]), 2)
        self.assertEqual(
            envelope["claims"][0]["evidence_ids"],
            [envelope["evidence"][0]["id"]],
        )
        self.assertEqual(
            envelope["evidence"][0]["source_class"],
            "government_registry",
        )
        self.assertEqual(
            envelope["evidence"][1]["source_class"],
            "company_owned",
        )
        self.assertEqual(envelope["operations"]["requests"], 3)

    def test_evidence_hash_covers_the_claim_span(self):
        envelope = contract.envelope_for(
            _profile(),
            run_id="run-1",
            started_at="2026-01-01T00:00:00Z",
            completed_at="2026-01-01T00:00:03Z",
        )
        expected = hashlib.sha256(
            b"Registered legal name: EXAMPLE AS"
        ).hexdigest()
        self.assertEqual(envelope["evidence"][0]["content_sha256"], expected)

    def test_evidence_carries_a_snapshot_reference(self):
        envelope = contract.envelope_for(
            _profile(),
            run_id="run-1",
            started_at="2026-01-01T00:00:00Z",
            completed_at="2026-01-01T00:00:03Z",
        )
        evidence = envelope["evidence"][0]
        snapshot = evidence["snapshot"]
        self.assertTrue(snapshot["id"].startswith("snap-"))
        self.assertEqual(
            snapshot["source_url"],
            "https://data.brreg.no/enheter/123456789",
        )
        self.assertEqual(snapshot["retrieved_at"], "2026-01-01T00:00:00Z")
        self.assertEqual(
            snapshot["content_sha256"], evidence["content_sha256"]
        )
        self.assertEqual(snapshot["extraction_method"], "api-json")

    def test_full_snapshot_hash_is_preferred_when_present(self):
        profile = _profile()
        profile["evidence"][0]["snapshot_sha256"] = "a" * 64
        envelope = contract.envelope_for(
            profile,
            run_id="run-1",
            started_at="2026-01-01T00:00:00Z",
            completed_at="2026-01-01T00:00:03Z",
        )
        evidence = envelope["evidence"][0]
        self.assertEqual(evidence["content_sha256"], "a" * 64)
        self.assertEqual(evidence["snapshot"]["content_sha256"], "a" * 64)

    def test_evidence_ids_are_stable_when_only_timestamps_change(self):
        first = _profile()
        second = json.loads(json.dumps(first))
        second["evidence"][0]["retrieved_at"] = "2026-02-01T00:00:00Z"
        second["facts"][0]["retrieved_at"] = "2026-02-01T00:00:01Z"
        second["retrieved_at"] = "2026-02-01T00:00:02Z"

        def ids(profile):
            envelope = contract.envelope_for(
                profile,
                run_id="run-1",
                started_at="2026-01-01T00:00:00Z",
                completed_at="2026-01-01T00:00:03Z",
            )
            return {
                claim["field"]: claim["evidence_ids"][0]
                for claim in envelope["claims"]
            }

        self.assertEqual(ids(first), ids(second))

    def test_evidence_ids_survive_evidence_reordering(self):
        first = _profile()
        second = json.loads(json.dumps(first))
        second["evidence"], second["facts"] = second["facts"], second["evidence"]

        def ids(profile):
            envelope = contract.envelope_for(
                profile,
                run_id="run-1",
                started_at="2026-01-01T00:00:00Z",
                completed_at="2026-01-01T00:00:03Z",
            )
            return {
                claim["field"]: claim["evidence_ids"][0]
                for claim in envelope["claims"]
            }

        self.assertEqual(ids(first), ids(second))

    def test_rerun_is_idempotent_modulo_timestamps(self):
        first = _profile()
        second = json.loads(json.dumps(first))
        second["evidence"][0]["retrieved_at"] = "2026-02-01T00:00:00Z"
        second["facts"][0]["retrieved_at"] = "2026-02-01T00:00:01Z"
        second["retrieved_at"] = "2026-02-01T00:00:02Z"

        def stable(profile):
            envelope = contract.envelope_for(
                profile,
                run_id="run-1",
                started_at="2026-01-01T00:00:00Z",
                completed_at="2026-01-01T00:00:03Z",
            )
            claims = [
                {
                    key: claim.get(key)
                    for key in (
                        "field",
                        "value",
                        "availability",
                        "evidence_ids",
                        "source_url",
                    )
                }
                for claim in envelope["claims"]
            ]
            evidence = [
                {
                    key: item.get(key)
                    for key in (
                        "id",
                        "source_url",
                        "source_class",
                        "content_sha256",
                        "claim_span",
                        "field",
                    )
                }
                for item in envelope["evidence"]
            ]
            return claims, evidence

        self.assertEqual(stable(first), stable(second))

    def test_synthesis_summarises_the_record(self):
        profile = _profile()
        profile["explanation"] = {
            "text": "EXAMPLE AS is a registered company.",
            "generated_by": "llm",
        }
        envelope = contract.envelope_for(
            profile,
            run_id="run-1",
            started_at="2026-01-01T00:00:00Z",
            completed_at="2026-01-01T00:00:03Z",
        )
        synthesis = envelope["synthesis"]
        self.assertEqual(
            synthesis["summary"], "EXAMPLE AS is a registered company."
        )
        self.assertEqual(synthesis["generated_by"], "llm")
        self.assertEqual(
            synthesis["claim_count"], len(envelope["claims"])
        )
        self.assertIn("1 field", synthesis["changes_summary"])

    def test_checked_empty_family_emits_not_available(self):
        profile = _profile()
        profile["availability"] = {"jobs": "not_available"}
        envelope = contract.envelope_for(
            profile,
            run_id="run-1",
            started_at="2026-01-01T00:00:00Z",
            completed_at="2026-01-01T00:00:03Z",
        )
        jobs = [claim for claim in envelope["claims"] if claim["field"] == "jobs"]
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["availability"], "not_available")
        self.assertIsNone(jobs[0]["value"])
        self.assertEqual(jobs[0]["evidence_ids"], [])

    def test_availability_does_not_duplicate_a_sourced_claim(self):
        profile = _profile()
        profile["availability"] = {"company_name": "not_available"}
        envelope = contract.envelope_for(
            profile,
            run_id="run-1",
            started_at="2026-01-01T00:00:00Z",
            completed_at="2026-01-01T00:00:03Z",
        )
        names = [
            claim
            for claim in envelope["claims"]
            if claim["field"] == "company_name"
        ]
        self.assertEqual(len(names), 1)
        self.assertEqual(names[0]["availability"], "available")

    def test_unknown_availability_state_is_ignored(self):
        profile = _profile()
        profile["availability"] = {"jobs": "maybe"}
        envelope = contract.envelope_for(
            profile,
            run_id="run-1",
            started_at="2026-01-01T00:00:00Z",
            completed_at="2026-01-01T00:00:03Z",
        )
        self.assertFalse(
            any(claim["field"] == "jobs" for claim in envelope["claims"])
        )

    def test_module_states_are_terminal_and_record_errors(self):
        envelope = contract.envelope_for(
            _profile(),
            run_id="run-1",
            started_at="2026-01-01T00:00:00Z",
            completed_at="2026-01-01T00:00:03Z",
        )
        states = {
            name: state["state"] for name, state in envelope["modules"].items()
        }
        self.assertEqual(states["registry"], "complete")
        self.assertEqual(states["roles"], "complete")
        self.assertEqual(states["financials"], "source_error")
        self.assertTrue(all(state in contract.TERMINAL_STATES for state in states.values()))

    def test_error_record_is_a_failed_terminal_envelope(self):
        envelope = contract.envelope_for(
            {"company_number": "123456789", "error": "ValueError: boom"},
            run_id="run-1",
            started_at="2026-01-01T00:00:00Z",
            completed_at="2026-01-01T00:00:03Z",
        )
        self.assertEqual(envelope["state"], "submission_error")
        self.assertEqual(envelope["run"]["terminal_status"], "failed")
        self.assertEqual(envelope["claims"], [])
        self.assertEqual(envelope["errors"][0]["code"], "profile_error")

    def test_budget_exhausted_record_is_budget_not_submission_error(self):
        envelope = contract.envelope_for(
            {
                "company_number": "123456789",
                "error": "RequestLimitExceeded: Outbound request limit exceeded: 2000",
                "budget_exhausted": True,
            },
            run_id="run-1",
            started_at="2026-01-01T00:00:00Z",
            completed_at="2026-01-01T00:00:03Z",
        )
        self.assertEqual(envelope["state"], "budget_exhausted")
        self.assertTrue(
            all(
                state["state"] in contract.TERMINAL_STATES
                for state in envelope["modules"].values()
            )
        )

    def test_budget_exhausted_state_is_terminal(self):
        self.assertIn("budget_exhausted", contract.TERMINAL_STATES)

    def test_network_source_failure_is_source_error_not_submission_error(self):
        envelope = contract.envelope_for(
            {
                "company_number": "123456789",
                "error": "ConnectionError: ('Connection aborted.', ConnectionResetError(10054, 'reset', None, 10054, None))",
            },
            run_id="run-1",
            started_at="2026-01-01T00:00:00Z",
            completed_at="2026-01-01T00:00:03Z",
        )
        self.assertEqual(envelope["state"], "source_error")
        self.assertEqual(
            envelope["modules"]["registry"]["state"], "source_error"
        )
        self.assertIn("source_error", contract.TERMINAL_STATES)

    def test_agent_defect_stays_submission_error(self):
        envelope = contract.envelope_for(
            {"company_number": "123456789", "error": "InvalidCompanyNumber: bad"},
            run_id="run-1",
            started_at="2026-01-01T00:00:00Z",
            completed_at="2026-01-01T00:00:03Z",
        )
        self.assertEqual(envelope["state"], "submission_error")


class ReportTests(unittest.TestCase):
    def _envelopes(self) -> list[dict]:
        return [
            contract.envelope_for(
                _profile(),
                run_id="run-1",
                started_at="2026-01-01T00:00:00Z",
                completed_at="2026-01-01T00:00:03Z",
            )
        ]

    def _report(self, envelopes, expected=1):
        return contract.build_report(
            envelopes,
            expected_count=expected,
            run_id="run-1",
            started_at="2026-01-01T00:00:00Z",
            completed_at="2026-01-01T00:00:03Z",
            runtime_ms=3000,
            requests=3,
            input_path="input.txt",
        )

    def test_exact_count_passes(self):
        report = self._report(self._envelopes())
        self.assertTrue(report["passed"])
        self.assertEqual(report["committed_claims"], 2)
        self.assertEqual(report["committed_evidence"], 2)

    def test_missing_company_fails(self):
        report = self._report(self._envelopes(), expected=2)
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["exact_expected_count"])
        self.assertFalse(report["checks"]["zero_silent_drops"])

    def test_duplicate_company_fails(self):
        envelope = self._envelopes()[0]
        report = self._report([envelope, envelope], expected=2)
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["unique_organisation_numbers"])


class CliTests(unittest.TestCase):
    def test_cli_writes_one_jsonl_envelope_and_a_passing_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.txt"
            output_path = root / "out.jsonl"
            profiles_path = root / "profiles.json"
            report_path = root / "report.json"
            input_path.write_text("123456789\n", encoding="utf-8")

            with patch.object(
                contract_cli,
                "process_companies",
                return_value=[_profile()],
            ):
                code = contract_cli.main(
                    [
                        "--input",
                        str(input_path),
                        "--output",
                        str(output_path),
                        "--profiles-output",
                        str(profiles_path),
                        "--report",
                        str(report_path),
                        "--expected-count",
                        "1",
                        "--minutes",
                        "0",
                        "--keyless",
                    ]
                )

            self.assertEqual(code, 0)
            envelopes = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(envelopes), 1)
            self.assertEqual(envelopes[0]["organisation_number"], "123456789")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertTrue(report["passed"])
            self.assertEqual(report["returned_count"], 1)


if __name__ == "__main__":
    unittest.main()
