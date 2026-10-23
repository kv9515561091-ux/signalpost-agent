"""A registered website is first-party only when the page names the company."""

import unittest

from src.identity import CompanyIdentity
from src.page import Page
from src.website import collect_social_links, verify_registered_page


def identity(**overrides):
    values = {
        "company_number": "989061593",
        "name": "-G- GROUP AS",
        "previous_names": [],
        "registered_website": "g-group.no",
        "registered_domain": "g-group.no",
    }
    values.update(overrides)
    return CompanyIdentity(**values)


def page(text="", *, title="", links=(), url="https://g-group.no/"):
    return Page(
        url=url,
        title=title,
        text=text,
        retrieved_at="2026-10-23T00:00:00+00:00",
        links=list(links),
    )


class WebsiteVerificationTests(unittest.TestCase):
    def test_organization_number_verifies_priority(self):
        check = verify_registered_page(
            identity(),
            page("Org.nr. 989 061 593"),
        )
        self.assertTrue(check.verified)
        self.assertEqual(check.basis, "organization_number")
        self.assertTrue(check.org_number_found)

    def test_exact_name_verifies_when_number_is_absent(self):
        check = verify_registered_page(
            identity(),
            page(title="G-Group"),
        )
        self.assertTrue(check.verified)
        self.assertEqual(check.basis, "exact_name")

    def test_previous_name_verifies(self):
        check = verify_registered_page(
            identity(previous_names=["Old Name AS"]),
            page("Old Name drives results across the Nordics."),
        )
        self.assertTrue(check.verified)
        self.assertEqual(check.basis, "previous_name")
        self.assertTrue(check.previous_name_found)
        self.assertFalse(check.exact_name_found)

    def test_page_without_company_evidence_is_unverified(self):
        check = verify_registered_page(
            identity(),
            page("Welcome to the Acme Group career site."),
        )
        self.assertFalse(check.verified)
        self.assertEqual(check.basis, "not_found")

    def test_short_names_are_not_enough(self):
        check = verify_registered_page(
            identity(name="AB AS"),
            page("AB"),
        )
        self.assertFalse(check.verified)

    def test_check_can_be_serialised(self):
        check = verify_registered_page(identity(), page("989061593"))
        data = check.as_dict()
        self.assertEqual(data["basis"], "organization_number")
        self.assertEqual(data["checked_at"], "2026-10-23T00:00:00+00:00")


class SocialLinkCollectionTests(unittest.TestCase):
    def test_only_matching_company_accounts_are_collected(self):
        page_obj = page(
            "G Group",
            links=[
                "https://www.linkedin.com/company/g-group/",
                "https://www.facebook.com/acmegroup/",
                "https://www.facebook.com/groups/123/",
                "/kontakt",
                "https://www.linkedin.com/company/g-group/",
            ],
        )
        links = collect_social_links(identity(), page_obj)
        self.assertEqual(
            links,
            [
                {
                    "platform": "linkedin",
                    "handle": "g-group",
                    "url": "https://www.linkedin.com/company/g-group/",
                }
            ],
        )

    def test_relative_social_links_are_resolved(self):
        links = collect_social_links(
            identity(),
            page("G Group", links=["/facebook"]),
        )
        self.assertEqual(links, [])

    def test_agency_account_is_not_inherited(self):
        links = collect_social_links(
            identity(),
            page(
                "G Group",
                links=["https://www.instagram.com/webagency/"],
            ),
        )
        self.assertEqual(links, [])


if __name__ == "__main__":
    unittest.main()
