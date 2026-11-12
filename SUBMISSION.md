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

The submitted commit is the repository `HEAD`. Record it at hand-off
rather than relying on a line in this file:

```bash
git rev-parse HEAD
```

CI is green on Python 3.10–3.13.

## 3. One command to run it

```bash
python run_contract.py \
  --input companies.txt \
  --output envelopes.jsonl \
  --profiles-output profiles.json \
  --report report.json \
  --expected-count 1000 \
  --keyless --workers 16 --minutes 35 \
  --html-output report/html
```

The `--html-output` flag additionally renders a self-contained static
site into `report/html`: a searchable `index.html` plus one page per
company (offline, no server, works on a phone). Every claim on every
page links to its source, so a reviewer can find a company, read its
profile and verify any fact without opening a JSON file. The flag is
optional and does not change the envelopes or the report.

The input holds one 9-digit organization number per line. The output is
one terminal envelope per company, in input order, plus a run report that
checks exact count, uniqueness, terminal entity/module states and silent
drops. Every claim carries source URL, retrieval time and evidence.

Runs of 500 or more automatically load the frozen Brreg identity export
(`data/brreg_bulk.csv.gz`) and underenhet export
(`data/brreg_underenheter.csv.gz`), downloading and caching each once when
absent. If the organizer supplies a registry snapshot, point the agent at
it instead of paying the one-time download with `AGENT_BULK=/path/to/enheter.csv.gz`
(and `AGENT_UNDERENHETER=` for workplaces); the run also degrades safely
to live identity requests if the download is unavailable, but no snapshot
at all at 1,000+ companies cannot fit the request ceiling. The underenhet
export adds registered workplaces without a live request per company. To
exercise the 100-company path with the deterministic registered-website
check:

```bash
python run_contract.py \
  --input data/competition_100_unique.txt \
  --output report/envelopes_100.jsonl \
  --profiles-output report/profiles_100.json \
  --report report/report_100.json \
  --expected-count 100 --keyless --workers 16 --minutes 35 \
  --bulk data/brreg_bulk.csv.gz \
  --underenheter data/brreg_underenheter.csv.gz \
  --verify-websites
```

It works **without any API key**. `--keyless` overrides keys present in
`.env`, so the result is reproducible on the organizer's machine. The
contract CLI deliberately has no `--resume`: every input is rebuilt and
rewritten each run. Prerequisites:

```bash
pip install -r requirements.txt
```

Keys are optional. With only `NVIDIA_API_KEY`, the agent extracts from the
registry-listed official website (search is skipped when no
`TINYFISH_API_KEY` is configured). With both keys it also discovers
additional verified sources. Credentials are read from `.env` but are
never printed. The rich daily path remains available through
`python run_daily.py`; the contract path above is the scored one.

## 4. Model and API details

| Component | Provider | Cost |
|---|---|---|
| Company identity, management, filed accounts | **Brønnøysundregistrene** (`data.brreg.no`) | Free, no key, no auth |
| Wikidata identity (QID, LEI, website) | **Wikidata Query Service** (`query.wikidata.org`) | Free, no key, one batch request |
| News developments | **Google News RSS** | Free, no key, no auth |
| Vacancies | **NAV Arbeidsplassen** (`arbeidsplassen.nav.no`) | Free, no key, frozen snapshot |
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

### Default contract run — keyless, bulk

Measured locally on the 1,000-number corpus:

| Metric | Value |
|---|---|
| Companies returned / expected | **1,000 / 1,000** |
| Claims / evidence | **30,573 / 30,500** |
| Outbound requests | **1,916 / 2,000** |
| Wall clock | **817 s** (~13.6 min, 16 workers) |
| Identities from frozen export | **997 / 1,000** |
| Registered workplaces | **821 across 634 companies** |
| Wikidata identities (QID) | **37** (3 with LEI) |
| Billable external | **$0** |

The 73-claim gap is deliberate: a company whose news lookup was
performed and found nothing publishes an explicit `news: not_available`
abstention with no evidence id. Abstentions never count toward recall
and are never dressed up as facts. The remaining 30,500 claims each
carry a source URL, retrieval time and an evidence record.

The cold-cache run above is the number the organizer will see on a
fresh clone: **1,916 of 2,000** requests leaves headroom so the tail is
never dropped at the ceiling. A warm-cache rerun reuses the content
cache and spends only a few hundred requests.

A 100-company smoke with the frozen snapshot, `--verify-websites` and
`--html-output` returned **100 / 100 complete**, **2,846 claims / 2,652
evidence**, 45 workplaces across 44 companies, 10 Wikidata identities
(2 with an LEI), and **345/2,000 requests** in ~224 s, plus a
searchable offline site with one page per company. Every module and
entity state is terminal; the report's invariants
(`exact_expected_count`, `unique_organisation_numbers`,
`all_entity_states_terminal`, `all_module_states_terminal`,
`zero_silent_drops`, `every_envelope_has_organisation_number`) all
pass. A fresh cache adds two export downloads; the figures above use
the cached snapshots.

### Optional enrichment path

Declared in calls, since search is the only billable external API. These
figures describe the optional TinyFish/NVIDIA path, not the keyless
contract default.

| | Per 100 companies |
|---|---|
| Brreg requests | 300 |
| Search calls | ~200 (2 per company; cached by query) |
| Total outbound requests | **1,616** (measured) |
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

A hundred-company rehearsal of the daily path at the 10-page cap ran the
cleaned 100-unique list end to end: **100 of 100 returned**, **3,536
claims** (1,064 of them extracted web facts, 10.6 per company), 3,447
evidence records, and **1,616/2,000 requests** in **20.2 minutes**.
80 companies carry at least one web fact, 1,061 web sources were read,
every fact quotes the page it came from, and every profile carries a
registry-backed explanation. Evidence ids and snapshots stay the same
across reruns, and a family that was probed and produced nothing is
published as an explicit availability claim rather than a silent gap:
89 such claims here (31 blocked, 50 not available, 8 failed). Five
companies publish a `careers_url`; none of this random hundred listed
openings on the day. A company with no web fact says why in
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

### Keyless external connectors

Three connectors add external intelligence with no API key, each gated
so it can never attach another company's data:

- **Wikidata** (`src/wikidata.py`) resolves the whole batch in one
  SPARQL request. Norwegian organisation numbers map to `wdt:P2333`; the
  query pulls the entity, LEI (`P1278`), website (`P856`) and employee
  count (`P1128`) with `GROUP_CONCAT` so multi-valued attributes cannot
  multiply rows. A company absent from the index records **no**
  availability state — a missing row is not a claim about the company,
  and 963 negative claims would be payload noise. Only positive identity
  facts are published.
- **News** (`src/news.py`) queries Google News RSS with a whole-phrase
  match on the folded legal name (legal form stripped). A story is
  attached only when that phrase appears in the title or summary and the
  item carries a real publication date; otherwise the family is recorded
  as `not_available`.
- **NAV vacancies** (`src/nav_jobs.py`) read a frozen snapshot so a
  vacancy search costs no per-company request. Postings are matched to
  the company by employer name, and the snapshot is only refreshed
  outside the scored run because the endpoint rate-limits aggressively.

The defaults (`AUTO_WEBSITE_CHECKS = 260`, `AUTO_NEWS_CHECKS = 90`,
Wikidata one batch call) are chosen so a cold 1,000-company run stays
under the ceiling. A measured cold run spent **1,916 / 2,000**; an
earlier 300/150 split hit 2,000 exactly and dropped the last seven
companies, which is why the defaults were lowered and a headroom test
(`tests/test_synthesis.py`) pins the arithmetic.

If the official batch grows past 1,000 companies (the contract says it
may reach 1,100), the automatic enrichment defaults shrink in proportion
to the room left by the registry spine so the extra ~130 registry
requests are taken out of enrichment rather than out of the tail
companies. Above 1,000 the selection reserves headroom explicitly; at
1,000 the proven defaults are unchanged. A run that nevertheless reaches
the ceiling emits an explicit `budget_exhausted` terminal state for the
affected companies — never a silent drop, and never `submission_error`,
which would read as a broken agent rather than an exhausted allowance.

## 6. Company profiles

At least 1,000 profiles are supplied.

- **All 1,000** are built registry-first: legal name and previous names,
  legal form, status, industry and code, business description, founding
  and registration dates, addresses, municipality, contact details,
  group and status flags, full management list including every director,
  registered workplaces, and multi-year filed accounts where they exist.
  Measured on the 1,000-number corpus: **1,000 / 1,000** resolved with a
  legal name, **919** carrying a management list, **511** carrying filed
  accounts, **821** registered workplaces across **634** companies, and
  **30,573** claims / **30,500** evidence. Every company and module
  state is terminal; there are no silent drops and no duplicate org
  numbers.
- **External enrichment** adds, where the budget reaches: **154**
  registry-listed websites verified by a page fetch, **56** companies
  with identity-matched social handles, **37** Wikidata identities
  (QID, and an LEI where Wikidata carries one), and identity-gated news
  developments. Every enriched fact quotes the page it came from.
- **Derived metrics** are computed from filed accounts, not restated:
  revenue growth, profit margin, equity ratio and revenue per employee
  for the latest year, plus a last-activity date. Years are sorted
  explicitly before the latest is chosen, so an out-of-order registry
  response cannot publish an older year as "latest".

Registry data is not a lesser record. The registry *is* the authoritative
source for everything it holds, and enrichment adds facts the registry
does not carry rather than correcting it. The optional web path still
supplies verified sources, quote-backed extraction, same-domain crawling
and a written narrative when keys are present.

The corpus is reproducible from the input list:

```
python run_contract.py \
  --input data/corpus_1000.txt \
  --output report/envelopes_1000.jsonl \
  --profiles-output report/profiles_1000.json \
  --report report/report_1000.json \
  --expected-count 1000 --keyless --workers 16 --minutes 35 \
  --bulk data/brreg_bulk.csv.gz \
  --underenheter data/brreg_underenheter.csv.gz
```

## 7. What the agent produces

Per company, one JSON profile containing:

- **Registry fields** — legal name and previous names, legal form,
  status, industry and code, business description, founding and
  registration dates, addresses, municipality, employee count, contact
  details, group status, bankruptcy and liquidation flags.
- **Registered workplaces** — parent-linked underenheter with name,
  address, municipality, industry, employee count and contact details,
  each carrying Brreg provenance.
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

The same run also renders a **self-contained static site** into the
directory named by `--html-output`: a searchable `index.html` (filter by
name or org number, state badges, facts/claims/evidence counts, works
offline on desktop and phone) plus one page per company that groups its
claims by information family and links every fact to its source. Registry
text is HTML-escaped on every page; `javascript:` URLs are dropped; each
envelope is reproducible from the input file. This is the surface a
reviewer uses to find a company, check its record and verify any fact
without parsing JSON.

## 8. Accuracy guarantees

These are what make the entry admissible; a wrong-company publication or
a fabricated financial value disqualifies it.

**Registry fields are never model-generated.** The registry owns them,
and the extraction schema does not include them — the model is never
offered the chance to restate a field the spine already holds.

**Workplaces are parent-linked registry records.** Each location comes
from the underenhet export and must name its parent organisation number.
The agent does not infer offices, headcount or addresses from a website.

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
when it equals the company name or a previous name after legal-form and
punctuation normalisation. Group pages, personal profiles, post/reel URLs
and parent-group or agency handles are rejected even when they appear in
a genuine footer; a registry-listed group domain does not make a group
handle first-party.

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
python -m unittest discover -s tests    # 770 tests
ruff check .                            # lint, enforced in CI
mypy                                    # 42 source files, no issues
```

CI runs the suite, ruff and mypy on Python 3.10–3.13, installs the
package and checks both console scripts, and verifies encoding integrity
on every source file.

Coverage is verified by an end-to-end test that replays recorded
registry, search and page responses through the real pipeline, so a
wiring regression fails in CI rather than being found by hand.