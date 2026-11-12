"""Tests for the keyless Google News RSS connector."""

from __future__ import annotations

import unittest

from src.news import (
    feed_url,
    match_news,
    news_facts,
    parse_feed,
    search_news,
)

FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <title>"Kongsberg Maritime" - Google News</title>
  <item>
    <title>Kongsberg Maritime vinner kontrakt i Brasil - NRK</title>
    <link>https://news.google.com/rss/articles/aaa</link>
    <pubDate>Sun, 09 Nov 2026 20:19:29 GMT</pubDate>
    <source url="https://nrk.no">NRK</source>
    <description>&lt;a href="x"&gt;Kongsberg Maritime&lt;/a&gt; har signert.</description>
  </item>
  <item>
    <title>Helt annen sak om fiskeri - VG</title>
    <link>https://news.google.com/rss/articles/bbb</link>
    <pubDate>Sat, 08 Nov 2026 10:00:00 GMT</pubDate>
    <source url="https://vg.no">VG</source>
    <description>Ingenting relevant.</description>
  </item>
  <item>
    <title>Kongsberg Maritime med ny fabrikk - DN</title>
    <link>https://news.google.com/rss/articles/ccc</link>
    <pubDate>Fri, 07 Nov 2026 08:00:00 GMT</pubDate>
    <source url="https://dn.no">DN</source>
    <description>Utvider.</description>
  </item>
  <item>
    <title>Kongsberg Maritime eldre sak - DN</title>
    <link>https://news.google.com/rss/articles/ddd</link>
    <pubDate>Mon, 01 Jan 2024 08:00:00 GMT</pubDate>
    <source url="https://dn.no">DN</source>
    <description>Gammelt.</description>
  </item>
</channel></rss>
"""


class FeedUrlTests(unittest.TestCase):
    def test_query_quotes_the_name_without_legal_form(self):
        # ``strip_legal_form`` also lowercases; the feed is
        # case-insensitive, so the query is sent in that form.
        url = feed_url("Kongsberg Maritime AS")
        self.assertIn("q=%22kongsberg%20maritime%22", url)
        self.assertIn("hl=nb", url)
        self.assertIn("gl=NO", url)
        self.assertIn("when%3A1y", url)

    def test_window_can_be_disabled(self):
        self.assertNotIn("when", feed_url("Acme AS", window=""))


class ParseFeedTests(unittest.TestCase):
    def test_items_are_parsed_newest_first(self):
        items = parse_feed(FEED.encode("utf-8"))
        self.assertEqual(len(items), 4)
        self.assertEqual(items[0].published, "2026-11-09")

    def test_publisher_suffix_is_removed_from_the_title(self):
        items = parse_feed(FEED.encode("utf-8"))
        self.assertEqual(
            items[0].title, "Kongsberg Maritime vinner kontrakt i Brasil"
        )
        self.assertEqual(items[0].publisher, "NRK")

    def test_markup_is_stripped_from_the_summary(self):
        items = parse_feed(FEED.encode("utf-8"))
        self.assertNotIn("<a", items[0].summary)
        self.assertIn("Kongsberg Maritime", items[0].summary)

    def test_broken_body_yields_nothing(self):
        self.assertEqual(parse_feed(b"not xml"), [])


class MatchNewsTests(unittest.TestCase):
    def test_only_items_naming_the_company_match(self):
        items = parse_feed(FEED.encode("utf-8"))
        matches = match_news(items, company_name="Kongsberg Maritime AS")
        self.assertEqual(len(matches), 2)
        for item in matches:
            self.assertIn("Kongsberg Maritime", item.title)

    def test_previous_names_also_match(self):
        items = parse_feed(FEED.encode("utf-8"))
        matches = match_news(
            items,
            company_name="Ukjent AS",
            previous_names=["Kongsberg Maritime"],
        )
        self.assertEqual(len(matches), 2)

    def test_short_names_never_match(self):
        items = parse_feed(FEED.encode("utf-8"))
        self.assertEqual(match_news(items, company_name="AS"), [])

    def test_partial_token_does_not_match(self):
        items = parse_feed(FEED.encode("utf-8"))
        self.assertEqual(match_news(items, company_name="Kongsberg Maritim"), [])

    def test_limit_caps_the_result(self):
        items = parse_feed(FEED.encode("utf-8"))
        matches = match_news(
            items, company_name="Kongsberg Maritime AS", limit=1
        )
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].published, "2026-11-09")


class SearchNewsTests(unittest.TestCase):
    def test_fetch_receives_the_feed_url(self):
        seen: list[str] = []

        def fetch(url: str) -> bytes:
            seen.append(url)
            return FEED.encode("utf-8")

        items = search_news("Kongsberg Maritime AS", fetch=fetch)
        self.assertEqual(len(items), 4)
        self.assertEqual(len(seen), 1)
        self.assertIn("news.google.com", seen[0])


class NewsFactsTests(unittest.TestCase):
    def test_fact_shape_is_publishable(self):
        items = parse_feed(FEED.encode("utf-8"))
        matches = match_news(items, company_name="Kongsberg Maritime AS")
        facts = news_facts(matches, retrieved_at="2026-11-09T21:00:00Z")
        self.assertEqual(len(facts), 2)
        fact = facts[0]
        self.assertEqual(fact["field"], "news")
        self.assertEqual(fact["source_type"], "news_source")
        self.assertEqual(fact["source_tier"], 5)
        self.assertEqual(fact["as_of"], "2026-11-09")
        self.assertEqual(fact["evidence_text"], fact["value"])
        self.assertTrue(fact["source_url"].startswith("https://"))

    def test_undated_items_are_not_published_as_news(self):
        undated = (
            b'<?xml version="1.0" encoding="UTF-8"?>\n'
            b'<rss version="2.0"><channel><title>Google News</title>\n'
            b'<item><title>Kongsberg Maritime wins contract</title>\n'
            b'<link>https://example.com/story</link><source>Dagens Naeringsliv</source>\n'
            b'<description>Kongsberg Maritime confirmed the deal.</description>'
            b'</item></channel></rss>'
        )
        items = parse_feed(undated)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].published, "")
        matches = match_news(items, company_name="Kongsberg Maritime AS")
        self.assertEqual(len(matches), 1)
        facts = news_facts(matches, retrieved_at="2026-11-09T21:00:00Z")
        self.assertEqual(facts, [])


class NewsSelectionTests(unittest.TestCase):
    def plan(self, inputs, records, limit):
        from src.contract_cli import _select_news_checks

        return _select_news_checks(inputs, records, limit)

    def test_zero_disables_the_lookup(self):
        self.assertEqual(self.plan([{"organisation_number": "1"}], {}, 0), set())

    def test_negative_means_every_company(self):
        self.assertIsNone(self.plan([{"organisation_number": "1"}], {}, -1))

    def test_short_names_are_never_selected(self):
        inputs = [{"organisation_number": "1"}, {"organisation_number": "2"}]
        records = {
            "1": {"company_name": "AS", "employee_count": 900},
            "2": {"company_name": "Kongsberg Maritime AS", "employee_count": 1},
        }
        self.assertEqual(self.plan(inputs, records, 5), {"2"})

    def test_largest_companies_come_first(self):
        inputs = [{"organisation_number": str(n)} for n in range(4)]
        records = {
            str(n): {
                "company_name": f"Company Number {n} AS",
                "employee_count": employees,
            }
            for n, employees in enumerate([5, 500, 50, 5000])
        }
        self.assertEqual(self.plan(inputs, records, 2), {"3", "1"})


if __name__ == "__main__":
    unittest.main()
