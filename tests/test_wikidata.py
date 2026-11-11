"""Tests for the batch Wikidata identity connector."""

from __future__ import annotations

import unittest

from src.wikidata import (
    WikidataEntity,
    identity_facts,
    normalise_org_numbers,
    parse_results,
    resolve,
    sparql_query,
)

PAYLOAD = {
    "results": {
        "bindings": [
            {
                "org": {"type": "literal", "value": "923609016"},
                "item": {
                    "type": "uri",
                    "value": "http://www.wikidata.org/entity/Q1776022",
                },
                "lei": {"type": "literal", "value": "OW6OFBNCKXC4US5C7523"},
                "website": {"type": "literal", "value": "https://equinor.com"},
                "employees": {"type": "literal", "value": "23000"},
            },
            {
                "org": {"type": "literal", "value": "982057892"},
                "item": {
                    "type": "uri",
                    "value": "http://www.wikidata.org/entity/Q3432029",
                },
                "lei": {"type": "literal", "value": ""},
                "website": {"type": "literal", "value": ""},
                "employees": {"type": "literal", "value": "35"},
            },
        ]
    }
}


class NormaliseTests(unittest.TestCase):
    def test_keeps_only_nine_digit_numbers_in_order(self):
        self.assertEqual(
            normalise_org_numbers(
                ["923 609 016", "12345", "982057892", "923609016", "x"]
            ),
            ["923609016", "982057892"],
        )


class QueryTests(unittest.TestCase):
    def test_query_lists_each_org_number_as_a_string(self):
        query = sparql_query(["923609016", "982057892"])
        self.assertIn('"923609016" "982057892"', query)
        self.assertIn("wdt:P2333", query)

    def test_query_uses_group_concat_so_rows_do_not_multiply(self):
        query = sparql_query(["923609016"])
        self.assertIn("GROUP_CONCAT", query)
        self.assertIn("GROUP BY", query)


class ParseTests(unittest.TestCase):
    def test_reads_identity_and_optional_attributes(self):
        found = parse_results(PAYLOAD)
        entity = found["923609016"]
        self.assertEqual(entity.qid, "Q1776022")
        self.assertEqual(entity.lei, "OW6OFBNCKXC4US5C7523")
        self.assertEqual(entity.website, "https://equinor.com")
        self.assertEqual(entity.employees, 23000)
        self.assertEqual(entity.url, "https://www.wikidata.org/wiki/Q1776022")

    def test_empty_group_concat_becomes_none(self):
        entity = parse_results(PAYLOAD)["982057892"]
        self.assertIsNone(entity.lei)
        self.assertIsNone(entity.website)
        self.assertEqual(entity.employees, 35)

    def test_a_row_without_an_item_is_skipped(self):
        self.assertEqual(
            parse_results(
                {"results": {"bindings": [{"org": {"value": "923609016"}}]}}
            ),
            {},
        )

    def test_a_row_that_carries_an_lei_wins_over_a_bare_qid(self):
        payload = {
            "results": {
                "bindings": [
                    {
                        "org": {"value": "923609016"},
                        "item": {"value": "http://www.wikidata.org/entity/Q1"},
                    },
                    {
                        "org": {"value": "923609016"},
                        "item": {"value": "http://www.wikidata.org/entity/Q2"},
                        "lei": {"value": "LEI-2"},
                    },
                ]
            }
        }
        self.assertEqual(parse_results(payload)["923609016"].qid, "Q2")

    def test_garbage_payload_is_empty(self):
        self.assertEqual(parse_results({}), {})
        self.assertEqual(parse_results({"results": {}}), {})


class ResolveTests(unittest.TestCase):
    def test_one_request_per_batch(self):
        calls: list[str] = []

        def fetch(query: str) -> dict:
            calls.append(query)
            return PAYLOAD

        found = resolve(
            [f"{index:09d}" for index in range(1500)],
            fetch=fetch,
            batch_size=1000,
        )
        self.assertEqual(len(calls), 2)
        # Only the two orgs the payload names resolve.
        self.assertEqual(set(found), {"923609016", "982057892"})

    def test_duplicates_are_collapsed_before_the_request(self):
        calls: list[str] = []

        def fetch(query: str) -> dict:
            calls.append(query)
            return {"results": {"bindings": []}}

        resolve(["923 609 016", "923609016"], fetch=fetch)
        self.assertEqual(calls[0].count('"923609016"'), 1)


class FactTests(unittest.TestCase):
    def test_qid_and_lei_are_published_with_provenance(self):
        entity = WikidataEntity(
            org_number="923609016",
            qid="Q1776022",
            lei="OW6OFBNCKXC4US5C7523",
        )
        facts = identity_facts(entity, retrieved_at="2026-11-11T00:00:00Z")
        fields = {fact["field"]: fact for fact in facts}
        self.assertEqual(fields["wikidata_id"]["value"], "Q1776022")
        self.assertEqual(fields["lei"]["value"], "OW6OFBNCKXC4US5C7523")
        for fact in facts:
            self.assertEqual(fact["source_type"], "wikidata")
            self.assertEqual(fact["source_tier"], 4)
            self.assertEqual(
                fact["retrieved_at"], "2026-11-11T00:00:00Z"
            )

    def test_no_lei_means_only_the_qid(self):
        entity = WikidataEntity(org_number="982057892", qid="Q3432029")
        fields = {
            fact["field"]
            for fact in identity_facts(entity, retrieved_at="t")
        }
        self.assertEqual(fields, {"wikidata_id"})


if __name__ == "__main__":
    unittest.main()
