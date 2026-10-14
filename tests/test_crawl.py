"""
Bounded same-domain crawl.

The planner is the only thing standing between "read more of the
company's own site" and "walk the internet". These tests are mostly
negative: what it must refuse matters more than what it picks.
"""

import unittest

from src.crawl import DEFAULT_MAX_CRAWL_LINKS, plan_crawl

HOME = "https://example.no"
DOMAIN = "example.no"


class BoundaryTests(unittest.TestCase):
    def test_other_domains_are_rejected(self):
        plan = plan_crawl(
            HOME,
            ["https://www.other.no/om-oss", "https://evil.example.com/about"],
            DOMAIN,
        )
        self.assertEqual(plan, [])

    def test_subdomains_are_rejected(self):
        # shop.example.no is a different registrable host from the
        # registered example.no and must not be followed blindly.
        plan = plan_crawl(HOME, ["https://shop.example.no/about"], DOMAIN)
        self.assertEqual(plan, [])

    def test_www_and_bare_are_the_same_domain(self):
        plan = plan_crawl("https://www.example.no", ["/om-oss"], DOMAIN)
        self.assertEqual(plan, ["https://www.example.no/om-oss"])

    def test_relative_links_resolve_against_the_page(self):
        plan = plan_crawl("https://example.no/some/deep/page", ["/om-oss"], DOMAIN)
        self.assertEqual(plan, ["https://example.no/om-oss"])

    def test_relative_links_keep_the_base_host(self):
        # A relative href off a www page resolves to that same www host.
        # Silently dropping to the apex would leave the verified domain,
        # and the apex may not serve the site at all.
        plan = plan_crawl("https://www.example.no", ["/om-oss"], DOMAIN)
        self.assertEqual(plan, ["https://www.example.no/om-oss"])

    def test_fragment_only_links_are_dropped(self):
        plan = plan_crawl(HOME, ["#main", "#"], DOMAIN)
        self.assertEqual(plan, [])

    def test_anchor_fragment_is_stripped(self):
        plan = plan_crawl(HOME, ["/om-oss#ledelse"], DOMAIN)
        self.assertEqual(plan, ["https://example.no/om-oss"])

    def test_non_page_schemes_are_dropped(self):
        plan = plan_crawl(
            HOME,
            ["mailto:a@example.no", "tel:+4712345678", "javascript:void(0)",
             "data:text/html,hi"],
            DOMAIN,
        )
        self.assertEqual(plan, [])

    def test_binary_assets_are_dropped(self):
        plan = plan_crawl(
            HOME,
            ["/logo.png", "/doc.pdf", "/app.js", "/style.css", "/feed.rss",
             "/photo.jpg", "/archive.zip", "/v.mp4"],
            DOMAIN,
        )
        self.assertEqual(plan, [])

    def test_self_is_never_replanned(self):
        plan = plan_crawl(HOME, [HOME, "https://example.no/"], DOMAIN)
        self.assertEqual(plan, [])

    def test_already_seen_are_skipped(self):
        plan = plan_crawl(
            HOME,
            ["/om-oss", "/kontakt"],
            DOMAIN,
            already_seen={"https://example.no/om-oss"},
        )
        self.assertEqual(plan, ["https://example.no/kontakt"])

    def test_duplicates_collapse(self):
        plan = plan_crawl(HOME, ["/om-oss", "/om-oss", "/om-oss"], DOMAIN)
        self.assertEqual(len(plan), 1)


class BoundsTests(unittest.TestCase):
    def _many(self, n):
        return [f"/side/page-{i}" for i in range(n)]

    def test_link_cap_is_enforced(self):
        plan = plan_crawl(HOME, self._many(50), DOMAIN, max_links=3)
        self.assertEqual(len(plan), 3)

    def test_zero_disables_the_crawl(self):
        self.assertEqual(plan_crawl(HOME, self._many(10), DOMAIN, max_links=0), [])

    def test_negative_disables_the_crawl(self):
        self.assertEqual(plan_crawl(HOME, self._many(10), DOMAIN, max_links=-1), [])

    def test_no_links_yields_nothing(self):
        self.assertEqual(plan_crawl(HOME, [], DOMAIN), [])

    def test_default_cap_exists(self):
        self.assertGreater(DEFAULT_MAX_CRAWL_LINKS, 0)
        self.assertLessEqual(DEFAULT_MAX_CRAWL_LINKS, 10)


class RankingTests(unittest.TestCase):
    def test_stable_pages_beat_press(self):
        plan = plan_crawl(
            HOME,
            ["/press/2026/launch", "/om-oss"],
            DOMAIN,
        )
        self.assertEqual(plan[0], "https://example.no/om-oss")

    def test_stable_pages_beat_people_pages(self):
        # A careers or leadership page is full of named individuals,
        # which this agent is not trying to publish.
        plan = plan_crawl(HOME, ["/personale/ledelse", "/tjenester"], DOMAIN)
        self.assertEqual(plan[0], "https://example.no/tjenester")

    def test_about_outranks_services(self):
        plan = plan_crawl(HOME, ["/tjenester", "/om-oss"], DOMAIN)
        self.assertEqual(plan[0], "https://example.no/om-oss")

    def test_unlabelled_pages_are_still_followed(self):
        # These sites are not built to a template. An allowlist would
        # drop every company whose paths differ, so unknown paths are
        # followed at a middling score rather than excluded.
        plan = plan_crawl(HOME, ["/whatever-this-is"], DOMAIN)
        self.assertEqual(plan, ["https://example.no/whatever-this-is"])

    def test_unknown_path_still_loses_to_about(self):
        plan = plan_crawl(HOME, ["/whatever-this-is", "/om-oss"], DOMAIN)
        self.assertEqual(plan[0], "https://example.no/om-oss")

    def test_ordering_is_stable_for_equal_scores(self):
        plan = plan_crawl(HOME, ["/bbb", "/aaa", "/ccc"], DOMAIN)
        self.assertEqual(plan, sorted(plan))

    def test_english_paths_are_recognised(self):
        plan = plan_crawl(HOME, ["/news/press-release", "/about-us"], DOMAIN)
        self.assertEqual(plan[0], "https://example.no/about-us")

    def test_negatives_are_evaluated_before_positives(self):
        # "/about-us/accessibility-statement" contains "about". Scoring
        # positives first would let every compliance page win on the
        # vocabulary it happens to share with a descriptive one.
        plan = plan_crawl(
            HOME, ["/about-us/accessibility-statement", "/about-us"], DOMAIN
        )
        self.assertEqual(plan, ["https://example.no/about-us"])

    def test_privacy_page_under_contact_is_excluded(self):
        plan = plan_crawl(
            HOME,
            ["/kontakt/artikler/personvern-og-debattregler", "/kontakt"],
            DOMAIN,
        )
        self.assertEqual(plan, ["https://example.no/kontakt"])

    def test_deep_paths_are_penalised(self):
        # Deep paths are archives and single press items, not descriptions.
        plan = plan_crawl(
            HOME,
            ["/om-oss/2021/06/press/old-archive-item"],
            DOMAIN,
        )
        self.assertEqual(plan, [])

    def test_shallow_about_beats_deep_about(self):
        plan = plan_crawl(HOME, ["/om-oss/2021/archive", "/om-oss"], DOMAIN)
        self.assertEqual(plan[0], "https://example.no/om-oss")


class RealisticNavTests(unittest.TestCase):
    def test_typical_nav_menu(self):
        links = [
            "/", "/no/om-oss", "/no/tjenester", "/no/kontakt",
            "/no/karriere", "/no/press", "/no/nyheter",
            "https://www.facebook.com/example", "https://twitter.com/example",
            "/no/personale",
        ]
        plan = plan_crawl(HOME, links, DOMAIN, max_links=4)

        self.assertTrue(all("facebook" not in u and "twitter" not in u for u in plan))
        self.assertEqual(plan[0], "https://example.no/no/om-oss")
        self.assertIn("https://example.no/no/tjenester", plan)
        self.assertIn("https://example.no/no/kontakt", plan)
        self.assertNotIn("https://example.no/no/press", plan)
        self.assertNotIn("https://example.no/no/personale", plan)

    def test_locale_prefixed_pages_survive(self):
        plan = plan_crawl(HOME, ["/en/about-us", "/no/om-oss"], DOMAIN)
        self.assertEqual(len(plan), 2)


if __name__ == "__main__":
    unittest.main()
