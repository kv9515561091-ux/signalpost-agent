# Submission — Signalpost Agent

Evidence-first Norwegian company profiles from official public sources.

---

## 1. Repository

```
https://github.com/kv9515561091-ux/signalpost-agent
```

## 2. Exact commit hash

```bash
git rev-parse HEAD
```

Current submission commit:

```
a6d4b72
```

CI is green on Python 3.10–3.13. Verify the exact hash at the moment of
submission rather than relying on this line.

## 3. One command to run it

```bash
python run_daily.py --input companies.txt --output profiles.json
```

The file holds one 9-digit organization number per line. One profile is
written per company, with registry lookup, web discovery, quote-verified
extraction, deterministic narrative, change detection against the
previous run, and a request budget that degrades rather than fails.

The same command works **without any API key**:

```bash
python run_daily.py --input companies.txt --output profiles.json --keyless
```

It then returns the registry spine plus a deterministic explanation for
every company and makes zero model calls. `--keyless` also overrides keys
present in `.env`, so the result is reproducible on the organizer's
machine. Prerequisites:

```bash
pip install -r requirements.txt
```

Keys are optional. With only `NVIDIA_API_KEY`, the agent extracts from the
registry-listed official website (search is skipped when no
`TINYFISH_API_KEY` is configured). With both keys it also discovers
additional verified sources. Credentials are read from `.env` but are
never printed.

`python run_daily.py` is resumable: re-running it after an interruption
skips companies already completed and reports only what changed. A run
that finished is refreshed in full on the next invocation: every company
is re-read and compared with the stored profile, so a quiet day is
reported as quiet only when the registry actually said so.

Verified keyless rehearsal: 10/10 companies, 0 failures, 10 deterministic
explanations, 0 fabricated facts, 22/2000 requests on the first pass and
20/2000 on the refresh, which reported `first seen 0` and `unchanged 10`.

The same mode is selected automatically when no key is configured, without
requiring `--keyless`; a two-company check reported 2/2 succeeded and
deterministic explanations, with any unverified registered site recorded
rather than trusted.

## 4. Model and API details

| Component | Provider | Cost |
|---|---|---|
| Company identity, management, filed accounts | **Brønnøysundregistrene** (`data.brreg.no`) | Free, no key, no auth |
| Web discovery | **TinyFish** search API | Billable per call — see §5 |
| Fact extraction + narrative | **NVIDIA NIM**, `nvidia/nemotron-3.5-super-vl-120b-a12b` | No per-token charge on the hosted build endpoint |
| Anthropic Claude | Supported, **not verified live** | — |

The agent speaks the Chat Completions shape directly over HTTP, so the
NVIDIA path adds no dependency. `anthropic` is an optional extra
(`pip install -e ".[anthropic]"`) and is mock-tested only; no Anthropic
key was available during development.

Extraction requests run at `temperature=0.0`, falling back to `1.0`
only if an endpoint rejects it. Model calls are paced to
`NVIDIA_RPM` (default 30). Measured, not assumed: 35 and 40 RPM drew
bursts of 429s and ran *slower* once the retries were counted, and 30
kept a fifty-company run within budget. The provider can still return
429 under load; the client pauses five and then ten seconds before
retrying, long enough for the rate window to turn over, rather than the
one- and two-second pauses that land in the same window. NVIDIA sends
no `Retry-After`, so pacing and backoff are the only controls.

## 5. Expected run costs

Declared in calls, since search is the only billable external API.

| | Per 100 companies |
|---|---|
| Brreg requests | 300 |
| Search calls | ~200 (2 per company; cached by query) |
| Total outbound requests | **~1,700–1,850** |
| LLM calls | ~600–800 |
| NVIDIA spend | $0 (hosted build endpoint) |
| **Billable external** | **~200 TinyFish calls** |

Enrichment depth is 8 sources / 10 pages per company. Two pages share
one model call, and each fact still names the page it came from and must
quote it verbatim. The next pair is fetched while the model reads the
current pair, so a page wait does not sit on the critical path.

Measured on a fifty-company run with the same shape as the daily test
(two pages per call, 12-page cap, serial fetches):

| Metric | Value |
|---|---|
| Requests | 750 (15 per company) |
| Wall clock | ~27 minutes (8 companies in flight) |
| Facts extracted | 838 (16.8 per company) |
| Companies with at least one fact | 50 / 50 |
| Narratives produced | 50 / 50 |
| **Facts missing evidence** | **0** |
| **Facts from an unverified (tier 7) source** | **0** |

With prefetch enabled, the same set finished in **21 minutes** (1,017
requests, including a re-search after the one-hour cache expired).

A hundred-company rehearsal of the daily path at the 10-page cap
completed 99 of 100 in **40.4 minutes** and 1,654 requests: 1,681
facts, 99 narratives, no fact without a verbatim quote, and no LLM
rate-limit failure. The one failure was a twenty-second Brreg read
timeout; because the run is resumable, a second pass retried only that
company (17 requests, 97 seconds) and finished **100 of 100, with 1,684
facts and 100 narratives** (100 records; the input list carried a
BOM-prefixed duplicate on the first line, fixed in the second rehearsal).
Primary run plus retry stayed inside both
the 2,000-request ceiling and the 45-minute limit. The 10-page cap
trades roughly 8% of facts against the twelve-page benchmark for that
headroom. A 138-company corpus build at 14 pages produced 616 facts
(4.5 per company) with zero unsupported facts, which is the
registry-first baseline the enriched subset is measured against.

A second rehearsal, after fixing the BOM, ran the cleaned 100-unique
list: **100 unique orgs**, 100 explanations, 97 with at least one
verified web fact, **0 errors**, **1,736 facts** across 1,133 web
sources. Two completed passes used 686 and 664 requests and each stayed
under the 45-minute and 2,000-request limits; the initial interrupted
pass supplied the first 30 profiles. The three profiles with no web
fact are two foreign entities and one association whose candidate
sources were all blocked or timed out; the reason is in
`retrieval_errors`, and the explanation falls back to the registry
spine.

`AGENT_REQUEST_LIMIT` sets the ceiling (default 2000). When the budget
drains, the agent reads fewer pages per company and past a floor falls
back to the registry spine alone — it degrades coverage rather than
exceeding the limit. A company whose enrichment was skipped says so in
`retrieval_errors`, so a reader can tell a profile with no facts
because the company has none from one that ran out of budget.

`AGENT_HTTP_TIMEOUT` and `AGENT_HTTP_ATTEMPTS` (defaults 20s / 2) let a
bulk corpus build trade patience for throughput without changing what a
scored daily run does.

## 6. Company profiles

At least 1,000 profiles are supplied.

- **All 1,000** are built registry-only: legal name and previous names,
  legal form, status, industry and code, business description, founding
  and registration dates, addresses, municipality, contact details,
  group and status flags, full management list including every director,
  and multi-year filed accounts where they exist. Measured on the
  delivered corpus: **1,000 / 1,000** resolved with a legal name,
  **920** carrying a filed management list, **511** carrying filed
  multi-year accounts, and **zero** duplicate org numbers or failed
  lookups.
- **138** of those are additionally enriched with verified web sources,
  quote-backed extraction, same-domain crawling and a written narrative.

Registry-only is not a lesser record. The registry *is* the authoritative
source for everything it holds, and enrichment adds facts the registry
does not carry rather than correcting it. The enriched subset shows
what the enrichment layer contributes.

Both files are reproducible with one command, and the corpus is
re-derivable from the input list:

```
python run_batch.py --input data/corpus_1000.txt --output data/registry_1000.json
python tools/verify_corpus.py data/registry_1000.json
```

## 7. What the agent produces

Per company, one JSON profile containing:

- **Registry fields** — legal name and previous names, legal form,
  status, industry and code, business description, founding and
  registration dates, addresses, municipality, employee count, contact
  details, group status, bankruptcy and liquidation flags.
- **Website and social** — `registered_website` (the registry value),
  `website` only when verified, `website_verified`/`website_verification`
  with the org-number or exact-name basis, and `social_links` whose
  handles match this company.
- **Management** — CEO, sole owner, board chair, board size, auditor,
  with the full role list including every director.
- **Financials** — multi-year filed accounts, consolidated-preferring,
  keyed by fiscal year with the reporting period retained.
- **Evidence** — every registry field carries its source URL,
  retrieval time and tier-bounded confidence.
- **Extracted facts** — from verified public pages, each with the
  verbatim sentence it came from.
- **Explanation** — a written narrative attributed to the model that
  produced it, with a deterministic fallback.
- **Gaps** — `retrieval_errors` and `extraction_rejected` name every
  source that failed and every claim that was refused, with the reason.
- **Changes** — what differs from the previous run, with before and
  after values.

## 8. Accuracy guarantees

These are what make the entry admissible; a wrong-company publication or
a fabricated financial value disqualifies it.

**Registry fields are never model-generated.** The registry owns them,
and the extraction schema does not include them — the model is never
offered the chance to restate a field the spine already holds.

**Every extracted fact must quote its source verbatim.** A quote that
cannot be found in the fetched page text is discarded. A quote citing a
different organization number is discarded regardless of what else it
says. When two pages share one model call, each fact must name its page
and is verified only against that page's text.

**Identity is verified before extraction, not after.** Sources are
classified into seven reliability tiers; listing pages, person pages and
foreign directory IDs are refused structurally. The org number settles
identity — a name never does on its own.

**A registered website is not automatically the company's own.** The
fetched page must carry the target organization number or the exact
registered name (including a previous name). A parent-group, brand or
web-agency page fails the check, is removed from `sources` and
`web_sources`, and cannot contribute facts, social links or news. The
reason is recorded in `retrieval_errors`; the unverified URL is retained
in `registered_website` for audit rather than published as `website`.

The keyless path performs this check too: it fetches the registered page
once and applies the same deterministic name/number test, with no model
call.

**Same-domain follow-ups are re-checked.** Each fetched follow-up must
also carry the organization number or exact name before its text is
extracted, so a footer link cannot lend a page trust it has not earned.
The crawl cannot widen the trust boundary.

**Social accounts must match the company.** A handle is published only
when it equals the company name, a previous name or the registered
domain label after legal-form and punctuation normalisation. Group pages,
personal profiles, post/reel URLs and parent-group or agency handles are
rejected even when they appear in a genuine footer.

**News needs a real publication date.** `recent_developments` is accepted
only when the quote contains a parseable publication date matching the
stated `published_date`. Navigation labels ("News", "See more"),
product/category pages, and "last updated"/"modified"/feed-updated
timestamps are rejected with a recorded reason.

**Every redirect hop is re-validated.** A discovered URL that redirects
to a private address is refused mid-chain; the guard applies to the URL
the agent ends up at, not the one it started from.

**Refusals are recorded, not hidden.** Rejected claims appear in
`extraction_rejected` with the reason.

## 9. Update behaviour

The daily test re-runs the same companies, so the agent compares each
profile against the one it stored last time and reports what moved. The
command refreshes a completed run and resumes only an interrupted one:
the state marks which companies finished, and if all of the requested
list is there, the next invocation rebuilds everything rather than
skipping the day. An interrupted refresh keeps the previous record as a
placeholder until its company is rebuilt, so a slow tail does not cost
the next run its comparison.

- Registry, contact, management, financial and extracted-fact changes
  are tagged separately, because a registry field changing is the
  company changing while a fact changing may only be a page rewritten.
- Every change carries the previous value, the new value, and the kind.
- Financial figures are keyed by fiscal year, so a newly filed year reads
  as one addition rather than making every later year look restated.
- Enhetsregisteret's per-entity record exposes no `endringsdato`, ETag
  or `Last-Modified`, so the comparison is made against stored profiles.
  The registry publishes a separate update feed
  (`/api/oppdateringer/enheter`); we query it to verify independently
  that a reported change is real. The Day 1 window contained 570
  registry updates, none of them in the 100.

## 10. Verification

```bash
python -m unittest discover -s tests    # test suite
ruff check .                            # lint, enforced in CI
mypy                                    # types, advisory in CI
```

CI runs the suite, ruff and mypy on Python 3.10–3.13, installs the
package and checks both console scripts, and verifies encoding integrity
on every source file.

Coverage is verified by an end-to-end test that replays recorded
registry, search and page responses through the real pipeline, so a
wiring regression fails in CI rather than being found by hand.