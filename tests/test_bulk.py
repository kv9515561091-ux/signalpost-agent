"""Frozen bulk registry snapshot loading."""

from __future__ import annotations

import gzip
import os
import tempfile
import unittest
from pathlib import Path

from src.bulk import ensure_bulk, ensure_underenheter, load_bulk, load_underenheter
from src.request_tracker import request_tracker

HEADER = (
    "organisasjonsnummer,navn,organisasjonsform.kode,"
    "organisasjonsform.beskrivelse,naeringskode1.kode,"
    "naeringskode1.beskrivelse,institusjonellSektorkode.kode,"
    "institusjonellSektorkode.beskrivelse,antallAnsatte,"
    "hjemmeside,epostadresse,telefon,forretningsadresse.adresse,"
    "forretningsadresse.postnummer,forretningsadresse.poststed,"
    "forretningsadresse.land,forretningsadresse.kommune,"
    "forretningsadresse.kommunenummer,sisteInnsendteAarsregnskap,"
    "registreringsdatoEnhetsregisteret,stiftelsesdato,konkurs,"
    "underAvvikling,erIKonsern,overordnetEnhet,aktivitet\n"
)

ROW = (
    "889640782,EKSEMPEL AS,AS,Aksjeselskap,62.010,"
    "Programmeringstjenester,2100,Private aksjeselskaper mv.,12,"
    "www.eksempel.no,post@eksempel.no,12345678,Testveien 1,"
    "0150,OSLO,Norge,OSLO,0301,2025,2020-01-02,2019-12-30,"
    "false,false,true,999999999,Utvikling av programvare\n"
)


class BulkTests(unittest.TestCase):
    def test_reads_requested_rows_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bulk.csv"
            path.write_text(HEADER + ROW, encoding="utf-8-sig")
            found, meta = load_bulk(
                path,
                ["889640782", "999999999"],
                retrieved_at="2026-11-03T00:00:00Z",
            )

        self.assertEqual(
            meta["missing"],
            ["999999999"],
        )
        self.assertEqual(meta["rows_scanned"], 1)
        record = found["889640782"]
        self.assertEqual(record["company_name"], "EKSEMPEL AS")
        self.assertEqual(record["employee_count"], 12)
        self.assertEqual(record["industry"], {"code": "62.010", "description": "Programmeringstjenester"})
        self.assertEqual(record["address"], "Testveien 1, 0150 OSLO, Norge")
        self.assertEqual(record["last_filed_annual_accounts"], "2025")
        self.assertIs(record["is_group_parent"], True)
        self.assertEqual(record["parent_company_number"], "999999999")
        self.assertEqual(record["source"], "https://data.brreg.no/enhetsregisteret/api/enheter/lastned/csv")
        self.assertEqual(len(meta["snapshot_sha256"]), 64)

    def test_reads_gzipped_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bulk.csv.gz"
            with gzip.open(path, "wt", encoding="utf-8-sig", newline="") as handle:
                handle.write(HEADER + ROW)
            found, meta = load_bulk(
                path,
                ["889640782"],
                retrieved_at="2026-11-03T00:00:00Z",
            )

        self.assertEqual(meta["selected"], 1)
        self.assertEqual(found["889640782"]["company_name"], "EKSEMPEL AS")

    def test_existing_snapshot_is_reused_without_a_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bulk.csv.gz"
            with gzip.open(path, "wb") as handle:
                handle.write(os.urandom(1_100_000))
            before = request_tracker.summary()["requests_used"]

            selected = ensure_bulk(path)

            self.assertEqual(selected, path)
            self.assertEqual(
                request_tracker.summary()["requests_used"],
                before,
            )

    def test_loads_underenheter_by_parent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "underenheter.csv"
            path.write_text(UNDER_HEADER + UNDER_ROW, encoding="utf-8")
            found, meta = load_underenheter(
                path,
                ["889640782", "999999999"],
                retrieved_at="2026-11-03T00:00:00Z",
            )

        self.assertEqual(meta["selected"], 1)
        self.assertEqual(meta["workplaces"], 1)
        self.assertEqual(
            found["889640782"][0]["name"],
            "EKSEMPEL AVDELING",
        )
        self.assertEqual(
            found["889640782"][0]["address"],
            "Avdelingsveien 2, 5003 BERGEN, Norge",
        )
        self.assertEqual(
            found["889640782"][0]["parent_company_number"],
            "889640782",
        )
        self.assertNotIn("999999999", found)

    def test_underenheter_snapshot_is_reused_without_a_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "underenheter.csv.gz"
            with gzip.open(path, "wb") as handle:
                handle.write(os.urandom(1_100_000))
            before = request_tracker.summary()["requests_used"]

            selected = ensure_underenheter(path)

            self.assertEqual(selected, path)
            self.assertEqual(
                request_tracker.summary()["requests_used"],
                before,
            )


UNDER_HEADER = (
    "organisasjonsnummer,navn,organisasjonsform.kode,"
    "organisasjonsform.beskrivelse,naeringskode1.kode,"
    "naeringskode1.beskrivelse,antallAnsatte,hjemmeside,"
    "epostadresse,telefon,beliggenhetsadresse.adresse,"
    "beliggenhetsadresse.postnummer,beliggenhetsadresse.poststed,"
    "beliggenhetsadresse.land,beliggenhetsadresse.kommune,"
    "beliggenhetsadresse.kommunenummer,"
    "registreringsdatoIEnhetsregisteret,overordnetEnhet,"
    "nedleggelsesdato\n"
)

UNDER_ROW = (
    "912345678,EKSEMPEL AVDELING,AAVD,Underenhet,62.010,"
    "Programmeringstjenester,8,www.eksempel.no/avdeling,"
    "avdeling@eksempel.no,87654321,Avdelingsveien 2,5003,BERGEN,"
    "Norge,BERGEN,4601,2021-03-04,889640782,\n"
)
