"""
End-to-end: discovery, identity, crawl, extraction, profile.

Every other test file in this suite is a unit test of one module with its
neighbours mocked out. That has a blind spot this file exists to close:
the wiring between those modules is only tested by running the thing.

The crawl is the clearest casualty. For several commits the crawl was
gated on a comparison between an ``Optional[str]`` and a ``str``. With no
registered website that evaluated False, the crawl silently did nothing,
and all 369 tests still passed because none of them drove discovery
through extraction with real page text. It was caught by hand, running
the CLI.

So this replays recorded HTTP responses -- registry, search, and the
real page HTML a crawl follows -- and asserts on the shape of the
finished profile rather than on internal calls:

* facts come from more than one page, which only holds if the crawler
  actually queued and read its follow-ups;
* every fact quotes text present in the page it claims to come from;
* crawled pages appear in ``web_sources`` carrying their inherited basis;
* nothing is published that repeats a registry field.

The model is stubbed. A recorded model response would have to be either
inconsistent with the recorded page (and so fail verification for the
wrong reason) or an assertion about which model happened to run. Instead
the stub quotes real sentences out of the page it was handed, so quote
verification does genuine work.

Regenerate the fixture with ``python tools/record_fixture.py``.
"""

import json
import re
import unittest
from pathlib import Path
from unittest.mock import patch

FIXTURE = Path(__file__).parent / "fixtures" / "pipeline.json"

with FIXTURE.open(encoding="utf-8") as handle:
    RECORDING = json.load(handle)

ORG = RECORDING["company_number"]
REGISTRY_BASE = "https://data.brreg.no/enhetsregisteret/api"
REGNSKAP_BASE = "https://data.brreg.no/regnskapsregisteret/regnskap"


def _sentence_candidates(text: str, limit: int = 4) -> list[str]:
    """Sentences long enough to be a meaningful fact.

    Drawn from the flattened page text so every returned quote is real by
    construction -- which is what lets the test exercise quote
    verification rather than bypass it.
    """
    flattened = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", flattened)

    out: list[str] = []
    for sentence in sentences:
        sentence = sentence.strip()
        # Long enough to say something, short enough to quote.
        if 90 <= len(sentence) <= 260:
            out.append(sentence)
        if len(out) >= limit:
            break
    return out


def _page_blocks(user: str) -> list[tuple[str, str]]:
    """Split a batch prompt into (page id, page text) pairs.

    A single-page prompt has no ``=== PAGE pN ===`` header, so this
    returns nothing and the caller falls back to reading the whole
    prompt.
    """
    blocks = []
    for part in user.split("=== PAGE ")[1:]:
        page_id, _, rest = part.partition(" ===")
        _, marker, content = rest.partition("--- PAGE CONTENT ---")
        blocks.append((page_id.strip(), content if marker else rest))
    return blocks


def _page_markup(entry: dict) -> str:
    """Rebuild a minimal page from recorded text and links.

    The recording stores flattened prose and the outbound links rather
    than raw HTML, because the first 18 KB of a corporate homepage is
    head, navigation and script -- flattening that yields menu labels and
    nothing quotable, which would leave extraction with no real sentence
    to verify against.

    Reassembling a minimal document lets ``html_to_text`` and
    ``extract_links`` both run for real, so the crawler still has genuine
    navigation to plan from. The HTML parser has its own unit tests; what
    this file exists to exercise is the wiring.
    """
    from html import escape

    body = [f"<title>{escape(entry.get('title', ''))}</title>"]
    body.append(f"<p>{escape(entry.get('text', ''))}</p>")
    for link in entry.get("links", []):
        body.append(f'<a href="{escape(link, quote=True)}">link</a>')
    return "<html><head>" + "".join(body[:1]) + "</head><body>" + "".join(body[1:]) + "</body></html>"


class ReplayResponse:
    """Just enough of ``requests.Response`` for the pipeline."""

    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self._text = text
        self.content = text.encode("utf-8")
        self.encoding = "utf-8"
        self.headers = {"Content-Type": "application/json"}
        self.url = ""

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON payload")
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"{self.status_code}")

    def iter_content(self, chunk_size=65536):
        yield self.content

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class RecordedTransport:
    """Serves recorded responses to whatever the pipeline asks for.

    Anything not in the recording 404s rather than reaching the network,
    so a test can never quietly pass by hitting a live endpoint.
    """

    def __init__(self):
        self.requested: list[str] = []
        # Trailing slashes are cosmetic and differ between the search
        # index and the registry record, so both sides are normalised
        # before lookup rather than guessing which form arrives.
        self._pages = {
            url.rstrip("/"): entry for url, entry in RECORDING["pages"].items()
        }

    def __call__(self, url, **kwargs):
        self.requested.append(url)

        if url.startswith(f"{REGNSKAP_BASE}/{ORG}"):
            return ReplayResponse(
                payload=RECORDING["registry"]["financials"]
            )

        if url.startswith(f"{REGISTRY_BASE}/enheter/{ORG}/roller"):
            return ReplayResponse(payload=RECORDING["registry"]["roles"])

        if url.startswith(f"{REGISTRY_BASE}/enheter/{ORG}"):
            return ReplayResponse(payload=RECORDING["registry"]["company"])

        if url.startswith("https://api.search.tinyfish.ai"):
            return ReplayResponse(
                payload={
                    "results": [
                        {
                            "url": hit["url"],
                            "title": hit.get("title", ""),
                            "snippet": hit.get("snippet", ""),
                        }
                        for hit in RECORDING["search"]
                    ]
                }
            )

        entry = self._pages.get(url.rstrip("/"))

        if entry is None or entry["kind"] == "error":
            return ReplayResponse(status_code=404, text="")

        return ReplayResponse(text=_page_markup(entry))


class RecordingTestCase(unittest.TestCase):
    def setUp(self):
        self.transport = RecordedTransport()
        patcher = patch("src.http.requests.get", self.transport)
        patcher.start()
        self.addCleanup(patcher.stop)

        # Without this every URL costs a real DNS lookup, and the suite
        # goes from two seconds to forty. The guard has its own tests.
        patcher = patch("src.netguard.resolve_is_public", lambda host: True)
        patcher.start()
        self.addCleanup(patcher.stop)

        # Keep the disk cache out of it, or a previous run's real pages
        # would satisfy a request the recording does not cover.
        from src import cache as cache_module

        cache_module.cache.clear()
        self.addCleanup(cache_module.cache.clear)

        for name in ("TINYFISH_API_KEY", "NVIDIA_API_KEY", "ANTHROPIC_API_KEY"):
            patcher = patch.dict("os.environ", {name: "replay-test"})
            patcher.start()
            self.addCleanup(patcher.stop)


class RegistrySpineTests(RecordingTestCase):
    def test_profile_builds_from_recorded_registry(self):
        from src.agent import build_profile

        profile = build_profile(ORG)

        self.assertEqual(profile.company_number, ORG)
        self.assertEqual(profile.company_name, "EQUINOR ASA")
        self.assertTrue(profile.evidence, "registry evidence should be recorded")
        self.assertTrue(profile.retrieved_at)

    def test_management_is_attached_from_the_recorded_roles(self):
        from src.agent import build_profile

        profile = build_profile(ORG)

        self.assertTrue(
            profile.ceo_name or profile.owner_name or profile.board_chair_name,
            f"expected a management role, got {profile.ceo_name!r}",
        )

    def test_financials_are_attached_when_recorded(self):
        from src.agent import build_profile

        profile = build_profile(ORG)

        if isinstance(RECORDING["registry"]["financials"], list):
            self.assertTrue(
                profile.financials,
                "the recording holds filed accounts, so the profile must too",
            )
        else:
            self.assertTrue(
                any("financials" in e for e in profile.retrieval_errors),
                f"no accounts recorded, so a gap must be reported: "
                f"{profile.retrieval_errors}",
            )


class DiscoveryTests(RecordingTestCase):
    def test_registered_website_is_seeded_first_party(self):
        from src.agent import build_profile

        profile = build_profile(ORG, include_web=True)

        domains = {s.domain for s in profile.web_sources}
        self.assertIn(
            "equinor.com",
            domains,
            "the registered website must be seeded as a source",
        )

    def test_org_number_survives_the_whole_pipeline(self):
        from src.agent import build_profile

        profile = build_profile(ORG, include_web=True)

        # Every published fact must belong to the target company. This is
        # the guarantee the whole project exists to provide, so it is
        # asserted on a real run rather than in isolation.
        for fact in profile.facts or []:
            self.assertIn(
                ORG,
                (fact.source_url or "") + (fact.evidence_text or ""),
                "a fact was published without a traceable origin",
            )


def _stub_result(facts):
    return type(
        "Result",
        (),
        {
            "data": {"facts": facts},
            "model": "stub",
            "provider": "stub",
            "finish_reason": "stop",
            "input_tokens": 0,
            "output_tokens": 0,
        },
    )()


def _run_enriched(**kwargs):
    """One full pipeline run against the recording.

    Shared across the class: each run replays six pages through
    discovery, crawling and extraction, and re-running it per assertion
    took the suite from two seconds to forty.
    """
    from src.agent import build_profile

    calls: list[str] = []

    def fake_complete_json(schema, *, system, user, **_):
        calls.append(user)
        blocks = _page_blocks(user)
        if blocks:
            return _stub_result(
                [
                    {
                        "field": "business_summary",
                        "value": sentence,
                        "quote": sentence,
                        "published_date": None,
                        "page": page_id,
                    }
                    for page_id, text in blocks
                    for sentence in _sentence_candidates(text)
                ]
            )
        return _stub_result(
            [
                {
                    "field": "business_summary",
                    "value": sentence,
                    "quote": sentence,
                    "published_date": None,
                }
                for sentence in _sentence_candidates(user)
            ]
        )

    with patch("src.extract.llm.is_available", return_value=True), patch(
        "src.extract.llm.unavailable_reason", return_value=None
    ), patch("src.extract.llm.complete_json", fake_complete_json):
        profile = build_profile(
            ORG,
            include_web=True,
            include_extraction=True,
            **kwargs,
        )

    return profile, calls


class ExtractionPipelineTests(RecordingTestCase):
    """The wiring the unit tests cannot see."""

    profile: object
    calls: list[str]

    @classmethod
    def setUpClass(cls):
        transport = RecordedTransport()
        with patch("src.http.requests.get", transport), patch(
            "src.netguard.resolve_is_public", lambda host: True
        ), patch.dict(
            "os.environ",
            {"TINYFISH_API_KEY": "replay", "NVIDIA_API_KEY": "replay"},
        ):
            from src import cache as cache_module

            cache_module.cache.clear()
            try:
                cls.profile, cls.calls = _run_enriched()
            finally:
                cache_module.cache.clear()

    def _run(self, **kwargs):
        return _run_enriched(**kwargs)

    def test_facts_are_published(self):
        self.assertTrue(self.calls, "extraction never reached the model")
        self.assertTrue(self.profile.facts, "no facts were published")

    def test_every_fact_quotes_its_page(self):
        profile = self.profile
        pages = RECORDING["pages"]

        for fact in profile.facts:
            entry = pages.get(fact.source_url)
            self.assertIsNotNone(
                entry,
                f"fact cites a page the recording does not have: {fact.source_url}",
            )
            haystack = entry.get("text") or ""
            self.assertIn(
                " ".join(fact.evidence_text.split()),
                " ".join(haystack.split()),
                "a quote was published that is not on the page it names",
            )

    def test_facts_come_from_more_than_one_page(self):
        """The assertion that would have caught the crawl regression.

        If the crawl stops queueing its follow-ups, this drops to one
        page and fails -- which no unit test did while the crawl was
        silently disabled.
        """
        profile = self.profile
        pages = {fact.source_url for fact in profile.facts}
        self.assertGreater(
            len(pages),
            1,
            "facts all came from one page; the crawl produced nothing",
        )

    def test_crawled_pages_are_recorded_with_an_inherited_basis(self):
        profile = self.profile
        crawled = [
            source
            for source in profile.web_sources
            if "same_domain" in (source.identity_basis or "")
        ]
        self.assertTrue(
            crawled,
            "crawled follow-ups must appear in web_sources with their basis",
        )
        for source in crawled:
            self.assertEqual(source.domain, "equinor.com")

    def test_crawled_pages_stay_on_the_registered_domain(self):
        profile = self.profile
        crawled = [
            source
            for source in profile.web_sources
            if "same_domain" in (source.identity_basis or "")
        ]
        for source in crawled:
            self.assertTrue(
                source.url.startswith("https://www.equinor.com"),
                f"crawl left the registered domain: {source.url}",
            )

    def test_no_registry_field_is_republished(self):
        profile = self.profile
        known = {
            profile.company_name,
            profile.website or "",
            profile.legal_form or "",
        }
        for fact in profile.facts:
            for value in known:
                if not value:
                    continue
                self.assertNotEqual(
                    fact.value.strip().lower(),
                    value.strip().lower(),
                    "a fact merely restates a registry field",
                )

    def test_crawl_cap_is_honoured(self):
        from src.agent import build_profile

        def fake_complete_json(schema, *, system, user, **kwargs):
            return type(
                "Result",
                (),
                {
                    "data": {"facts": []},
                    "model": "stub",
                    "provider": "stub",
                    "finish_reason": "stop",
                    "input_tokens": 0,
                    "output_tokens": 0,
                },
            )()

        with patch("src.extract.llm.is_available", return_value=True), patch(
            "src.extract.llm.unavailable_reason", return_value=None
        ), patch("src.extract.llm.complete_json", fake_complete_json):
            profile = build_profile(
                ORG,
                include_web=True,
                include_extraction=True,
                max_crawl_links=0,
            )

        crawled = [
            source
            for source in profile.web_sources
            if "same_domain" in (source.identity_basis or "")
        ]
        self.assertEqual(crawled, [], "max_crawl_links=0 must disable the crawl")


class FixtureIntegrityTests(unittest.TestCase):
    """The fixture must stay capable of catching things."""

    def test_fixture_covers_the_crawl(self):
        home = "https://www.equinor.com"
        self.assertIn(home, RECORDING["pages"])
        follow_ups = [
            url
            for url in RECORDING["pages"]
            if url.startswith(home) and url != home
        ]
        self.assertTrue(
            follow_ups, "no same-domain follow-up recorded; the crawl is untestable"
        )

    def test_recorded_pages_carry_navigation(self):
        home = RECORDING["pages"]["https://www.equinor.com"]
        self.assertTrue(
            home.get("links"),
            "homepage has no recorded links; the crawl cannot plan",
        )

    def test_recorded_pages_carry_quotable_prose(self):
        # If this drops, extraction has nothing to verify against and the
        # pipeline test would pass for the wrong reason.
        for url, entry in RECORDING["pages"].items():
            if entry["kind"] != "page":
                continue
            self.assertTrue(
                len(entry.get("text", "")) > 500,
                f"{url} recorded almost no prose",
            )


if __name__ == "__main__":
    unittest.main()
