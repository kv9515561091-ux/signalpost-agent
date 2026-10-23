"""Social handles must name the company, not its parent or web agency."""

import unittest

from src.social import (
    account_from_url,
    is_social_url,
    social_handle_matches,
    social_platform,
)


class PlatformTests(unittest.TestCase):
    def test_known_platforms_are_recognised(self):
        cases = {
            "https://www.facebook.com/equinor/": "facebook",
            "https://fb.com/equinor": "facebook",
            "https://www.linkedin.com/company/equinor/": "linkedin",
            "https://www.instagram.com/equinor/": "instagram",
            "https://x.com/equinor": "x",
            "https://twitter.com/equinor": "twitter",
            "https://www.youtube.com/@equinor": "youtube",
            "https://youtu.be/abc": "youtube",
            "https://www.tiktok.com/@equinor": "tiktok",
        }
        for url, platform in cases.items():
            self.assertEqual(social_platform(url), platform, url)

    def test_non_social_urls_are_not_social(self):
        self.assertFalse(is_social_url("https://www.equinor.com/"))
        self.assertFalse(is_social_url("https://facebook.com.evil.example/"))
        self.assertFalse(is_social_url("not a url"))


class AccountTests(unittest.TestCase):
    def test_company_page_yields_a_handle(self):
        self.assertEqual(
            account_from_url("https://www.facebook.com/Equinor/"),
            ("facebook", "Equinor"),
        )
        self.assertEqual(
            account_from_url("https://www.linkedin.com/company/equinor/"),
            ("linkedin", "equinor"),
        )
        self.assertEqual(
            account_from_url("https://www.youtube.com/@equinor"),
            ("youtube", "equinor"),
        )

    def test_group_and_personal_profiles_are_not_company_accounts(self):
        for url in (
            "https://www.facebook.com/groups/130557173661377/",
            "https://www.facebook.com/people/Equinor/123/",
            "https://www.linkedin.com/in/ola-nordmann/",
            "https://www.instagram.com/p/abc123/",
            "https://x.com/intent/post",
            "https://www.youtube.com/channel/UC123",
        ):
            self.assertIsNone(account_from_url(url), url)

    def test_post_urls_are_not_account_links(self):
        for url in (
            "https://x.com/equinor/status/123",
            "https://www.tiktok.com/video/123",
        ):
            self.assertIsNone(account_from_url(url), url)

    def test_facebook_pages_path_is_understood(self):
        self.assertEqual(
            account_from_url("https://www.facebook.com/pages/Equinor/123"),
            ("facebook", "Equinor"),
        )


class MatchTests(unittest.TestCase):
    def test_handle_matches_name_after_legal_form_and_punctuation(self):
        self.assertTrue(
            social_handle_matches(
                "https://www.linkedin.com/company/g-group/",
                company_name="-G- GROUP AS",
            )
        )

    def test_previous_name_is_accepted(self):
        self.assertTrue(
            social_handle_matches(
                "https://www.facebook.com/oldname/",
                company_name="New Name AS",
                previous_names=("Old Name AS",),
            )
        )

    def test_registered_domain_label_is_accepted(self):
        self.assertTrue(
            social_handle_matches(
                "https://www.instagram.com/equinor/",
                company_name="Equinor ASA",
                registered_website="www.equinor.com",
            )
        )

    def test_handle_with_legal_suffix_is_accepted(self):
        self.assertTrue(
            social_handle_matches(
                "https://twitter.com/EquinorASA",
                company_name="Equinor ASA",
            )
        )

    def test_parent_group_handle_is_rejected(self):
        self.assertFalse(
            social_handle_matches(
                "https://www.facebook.com/acmegroup/",
                company_name="Acme Norge AS",
                registered_website="acme.no",
            )
        )

    def test_short_or_missing_handles_are_rejected(self):
        self.assertFalse(
            social_handle_matches(
                "https://www.facebook.com/ab/",
                company_name="AB AS",
            )
        )
        self.assertFalse(
            social_handle_matches(
                "https://www.facebook.com/groups/123/",
                company_name="Group AS",
            )
        )


if __name__ == "__main__":
    unittest.main()
