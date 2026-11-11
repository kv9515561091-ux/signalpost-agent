"""Tests for dated activity read from a first-party page."""

from __future__ import annotations

import unittest
from datetime import date

from src.activity import dated_items, find_date


class FindDateTests(unittest.TestCase):
    def test_norwegian_numeric_date(self):
        found = find_date("Ny kontrakt signert 09.11.2026")
        self.assertEqual(found[0], date(2026, 11, 9))

    def test_norwegian_worded_month(self):
        found = find_date("Ny kontrakt signert 9. november 2026")
        self.assertEqual(found[0], date(2026, 11, 9))

    def test_abbreviated_month(self):
        found = find_date("Lansering 3. des 2025")
        self.assertEqual(found[0], date(2025, 12, 3))

    def test_iso_date(self):
        found = find_date("published 2026-11-09T20:19:29+01:00")
        self.assertEqual(found[0], date(2026, 11, 9))

    def test_impossible_date_is_rejected(self):
        self.assertIsNone(find_date("31.02.2026"))

    def test_no_date(self):
        self.assertIsNone(find_date("Vi leverer kvalitet"))


class DatedItemsTests(unittest.TestCase):
    TODAY = date(2026, 11, 10)

    def test_headline_before_the_date_is_used(self):
        text = "\n".join(
            [
                "Nyheter",
                "Kongsberg Maritime vinner ny kontrakt i Brasil",
                "09.11.2026",
            ]
        )
        items = dated_items(text, today=self.TODAY)
        self.assertEqual(len(items), 1)
        self.assertEqual(
            items[0]["title"],
            "Kongsberg Maritime vinner ny kontrakt i Brasil",
        )
        self.assertEqual(items[0]["date"], "2026-11-09")

    def test_headline_on_the_same_line_is_used(self):
        items = dated_items(
            "09.11.2026 Ny kontrakt med Forsvaret signert",
            today=self.TODAY,
        )
        self.assertEqual(len(items), 1)
        self.assertEqual(
            items[0]["title"], "Ny kontrakt med Forsvaret signert"
        )

    def test_evidence_text_is_verbatim_page_text(self):
        line = "09.11.2026 Ny kontrakt med Forsvaret signert"
        items = dated_items(line, today=self.TODAY)
        self.assertIn(items[0]["evidence_text"], line)

    def test_old_items_are_ignored(self):
        text = "Gammel sak\n01.01.2020\n"
        self.assertEqual(dated_items(text, today=self.TODAY), [])

    def test_future_items_are_ignored(self):
        text = "Planlagt sak\n01.06.2027\n"
        self.assertEqual(dated_items(text, today=self.TODAY), [])

    def test_privacy_and_cookie_lines_are_ignored(self):
        text = "Personvernerklæring oppdatert 01.11.2026\n"
        self.assertEqual(dated_items(text, today=self.TODAY), [])

    def test_short_labels_are_ignored(self):
        text = "Nyhet\n09.11.2026\n"
        self.assertEqual(dated_items(text, today=self.TODAY), [])

    def test_items_are_newest_first_and_capped(self):
        text = "\n".join(
            [
                "Første sak om ny fabrikk i Trondheim",
                "01.01.2026",
                "Andre sak om nytt anlegg i Bergen",
                "01.06.2026",
                "Tredje sak om oppkjøp av selskap",
                "01.10.2026",
                "Fjerde sak om nytt kontor i Oslo",
                "01.11.2026",
            ]
        )
        items = dated_items(text, today=self.TODAY)
        self.assertEqual(len(items), 3)
        self.assertEqual(
            [item["date"] for item in items],
            ["2026-11-01", "2026-10-01", "2026-06-01"],
        )

    def test_duplicate_headline_is_published_once(self):
        text = "\n".join(
            [
                "Ny kontrakt med Forsvaret signert",
                "01.10.2026",
                "Ny kontrakt med Forsvaret signert",
                "01.10.2026",
            ]
        )
        items = dated_items(text, today=self.TODAY)
        self.assertEqual(len(items), 1)

    def test_empty_page(self):
        self.assertEqual(dated_items("", today=self.TODAY), [])


if __name__ == "__main__":
    unittest.main()
