# Signalpost Agent

Builds a **verified, evidence-backed company profile** from a Norwegian
organization number.

```bash
python -m src.agent 989061593
```

The agent is not a search chatbot. It resolves an organization number to a
canonical identity at the Norwegian Business Register, then treats
everything else as a *claim* that must prove itself. Each published fact
carries its source URL, a retrieval timestamp, the date the fact itself
describes, and a confidence bounded by how authoritative that source is.

> Fewer verified facts beat many plausible ones.

---

## Quick start

```bash
pip install -r requirements.txt
# or, for the console scripts and tool config:
pip install -e ".[dev]"

python -m src.agent --check                         # is my setup right?
python -m src.agent 989061593                      # registry spine only
python -m src.agent 989061593 --narrate            # + written explanation
python -m src.agent 989061593 --web                # + verified public sources
python -m src.agent 989061593 --extract --narrate  # everything
python -m src.agent 989061593 --json -v            # raw profile + provenance
python -m src.agent --clear-cache                  # drop cached responses

# Batch. Registry-only by default; enrichment is opt-in because every
# page costs a model call.
python run_batch.py
python run_batch.py --web --extract --narrate
python run_batch.py --extract --max-sources 4 --max-pages 6
python run_batch.py --no-llm                       # deterministic, free
python run_batch.py --resume                       # continue an interrupted run
python -m src.agent 989061593 --extract --max-crawl-links 0   # no crawl

python -m unittest discover -s tests               # 440 tests
ruff check .                                         # lint
mypy                                                 # types (advisory)
```

The enrichment flags — `--web`, `--extract`, `--narrate`, `--no-llm`,
`--max-sources`, `--max-pages`, `--max-crawl-links` — are identical in
both, deliberately. A separate vocabulary per entry point is how
`--extract` ends up meaning something different in each. `tests/test_cli.py`
asserts the shared set stays shared and that the differences are only the
intentional ones.

What differs, on purpose: `--input`, `--output` and `--delay` exist only
for batches; `--check`, `--clear-cache`, `--json` and `-v` only for a
single company.

Optional LLM features use either provider. Put one key in `.env`:

```bash
ANTHROPIC_API_KEY=sk-ant-...     # Claude; needs: pip install anthropic
NVIDIA_API_KEY=nvapi-...         # NVIDIA NIM; nothing to install
```

An Anthropic key wins if both are set; `LLM_PROVIDER` overrides the
choice (`anthropic` | `nvidia` | `none`). Without a key the agent runs
unchanged — see [Optional LLM](#optional-llm).

Configured credentials always beat a merely discoverable one. The
Anthropic SDK resolves an on-disk profile with no environment variable
set, so without this rule a machine with Claude Code installed would use
Claude even after NVIDIA had been configured. `--check` prints which
provider is actually active.

### Caching

Registry responses are cached under `data/cache/` (gitignored) for six
hours, so repeat runs cost nothing:

```
COLD: Cache: 0 hit(s), 3 written | Requests used: 3/2000
WARM: Cache: 3 hit(s), 0 written | Requests used: 0/2000
```

Search results are cached too, but for **one hour** rather than six —
page wording and search ranking drift, while a company's legal identity
does not. An hour still covers a whole batch run.

> The search cache is keyed on the *expanded* URL, query string
> included. Every search goes to the same endpoint with the query in the
> query string, so keying on the bare endpoint would serve one company's
> results to every other company — a silent, expensive, wrong-answer bug
> rather than a visible failure. `tests/test_search_cache.py` asserts
> isolation first and cost second.

Measured: a repeat query drops from 2.54 s / 1 request to 0.03 s / 0.

A cache entry past its time-to-live is **not** silently reused. It is
served only when revalidation fails, and the profile then carries
`served_stale: true` plus a retrieval gap saying so. A 404 is never
replaced by a stale body — "no such record" is the current truth.

### Rate limiting

NVIDIA's hosted endpoint enforces a request-per-minute ceiling that it
does not publish, and returns 429 with **no `Retry-After`**, so a client
cannot react intelligently to being throttled. Requests are paced up
front instead of discovering the limit by tripping it.

| | |
|---|---|
| `NVIDIA_RPM` | Requests per minute. Default **30**, below the observed ~40 RPM limit, so a run has headroom rather than running exactly at the edge |
| Inter-company `--delay` | A different thing. It spaces *companies*; the limiter spaces *model calls* within one company |

Once extraction is on, a ten-company batch makes roughly 60 model calls,
so this is load-bearing rather than decorative. Backoff on 429 still
applies as a second line of defence.

Set `NVIDIA_RPM=0` to disable pacing — useful for offline tests, which
otherwise sleep between mocked calls.

---

## How it works

```
org.nr ──► MOD-11 ──► Brreg identity ──┬─► /roller    management, board, auditor
                 │                     ├─► /regnskap  multi-year filed accounts
                 │                     └─► ~35 fields  purpose, NACE, status, contact
                 ▼
           web discovery ──► identity gate ──► source tiering
                 ▼
           page fetch ──► LLM extraction ──► verbatim-quote verification
                 ▼
           conflict reconciliation ──► explanation ──► profile
```

### 1. Input validation before any network call

Organization numbers carry a MOD-11 check digit, so typos are rejected
locally instead of costing an API call. Verified against 500 consecutive
live Enhetsregisteret records.

### 2. The registry is the spine

`Brønnøysundregistrene` supplies identity, management and filed financials
as **Tier-1 evidence**, with no authentication and no cost. Everything
discovered later is checked against this identity.

Three details of the accounts API are easy to get wrong, and each was
verified against Equinor's published FY2025 annual report:

| Pitfall | Consequence if missed |
|---|---|
| Prefer `regnskapstype: KONSERN` | `SELSKAP` is parent-only — Equinor revenue reads **67.9bn instead of 106.5bn** |
| Read `valuta` | Reporting currency is not always NOK; Equinor files in **USD** (~11× error) |
| Tolerate absent optional lines | `loennskostnad` is missing under large-company presentation plans |

### 3. Identity verification, not name matching

Company names are reused, re-branded and translated constantly. The
organization number settles identity; a name never does on its own.

Sources are tiered 1–7, and structural traps are caught before any text
is read:

- **Listing pages** — `proff.no/bransjesøk?q=989061593` carries the org
  number in its *query string*. It is a list of many companies, so its
  text belongs to an unknown subset. Refused.
- **Person pages** — `firmadatabasen.no/person/...` describes a human, not
  the company. Refused.
- **Foreign identifiers** — `gulesider.no/.../125157773/` has its own
  9-digit directory ID. Only the *target* number counts.

### 4. Extraction the agent can check

When an LLM reads a verified page, it does not get to decide what is true.
Every proposed fact must arrive with a `quote` copied verbatim from the
page:

```
quote absent from the page      → discarded
quote cites another org number  → discarded
field owned by the registry     → never even offered to the model
confidence above source tier    → capped
```

This turns hallucination from invisible into detectable. Facts the model
proposes but cannot substantiate are recorded in `extraction_rejected`
with the reason, rather than silently vanishing.

Precision rules layered on top, each of which discards something a model
will happily produce:

- **Registry restatement.** A directory reprinting a value Brreg already
  published is not a new discovery, so it is dropped as redundant.
- **Cross-page deduplication.** The same statement from six sources is
  published once.
- **Business-only.** Personal biography is refused in code, not by
  instruction. See below.
- **Official website first.** The registered domain is seeded as a
  Tier-2 source ahead of search results, since it is the strongest thing
  the agent can find.
- **Deterministic temperature.** Extraction asks for `temperature=0.0`
  and falls back to `1.0` only if the endpoint rejects it.

### 5. A bounded crawl inside the trust boundary

A homepage is a thin basis for a profile; the About and Services pages
carry the durable facts. So a verified first-party page may enqueue
follow-ups from **its own domain**.

This is safe where arbitrary search is not. Every URL sits on the domain
the registry already vouched for, so it inherits the homepage's
verification rather than earning its own — the crawler never widens the
trust boundary, it only walks inside one that has been proven. The walk
is depth one and hard-capped (`--max-crawl-links`, default 5), so it
cannot run away into a sitemap.

Link choice prefers *stable* pages and rejects press, personnel and
compliance pages. Those are not merely ranked last: they score negative
and are dropped, because every followed page costs a model call.

Negatives are evaluated **before** positives, which is not a detail.
`/about-us/accessibility-statement` contains "about" — with positives
checked first, every compliance page wins on the vocabulary it happens to
share, and TINE's nav still surfaced a privacy statement until the order
was reversed.

Unknown paths are followed at a middling score rather than excluded. These
sites are not built to a template, and an allowlist would silently return
nothing for every company whose paths differ.

Crawled pages are recorded in `web_sources` with a `+same_domain` basis,
so provenance stays visible.

Measured on Equinor: 3 facts on one page → 12 after raising the caps →
**26 across three pages** — headcount, geographic footprint, portfolio and
market position, rather than press releases.

### 5b. Personal biography is refused, not prompted against

The prompt tells the model not to report someone's family, health or
personal history. That instruction is the weakest link in the pipeline:
the quote check proves a sentence exists on the page, not that the
sentence is *about the company*. A page mixing a founder's CV with a
company description passes identity, passes the quote check, and then
contributes the CV.

So `_is_personal_content` refuses it in code, with the reason recorded in
`extraction_rejected` like every other refusal.

False positives are the real risk — a filter that eats legitimate company
facts is worse than no filter, because it looks like working. These all
survive, and each is a test:

```
A family-owned Norwegian manufacturer.          → kept
Healthcare services for hospitals and clinics.  → kept
Products for children's hospitals.              → kept
Hobbies are our business.                        → kept
```

The test is a *distinct* personal signal plus a density floor, not
density alone. Density was tried first and refused real biography: "He
is married to Anna and they have three children" is only 2 personal
words across 10, because pronouns carry the sentence while the actual
signal — "children" — is one word. Tuning the threshold down to catch it
then put "our founder's wife Anna led the charity" at 1 word in 10.

### 5c. Every redirect hop is re-checked

Discovered URLs come from a search index, so they are attacker-influenced.
The guard answers "may the agent open a connection to this host?" — and
it must be asked again at every hop, not only for the URL supplied.

Delegating redirects to the HTTP library meant it was asked once. A
discovered `https://company.example` that 302s to
`http://169.254.169.254/latest/meta-data/` passed the check and reached
cloud instance metadata anyway. Validating the URL the agent decided to
check is worthless when the request it never made is the one that goes
out.

Redirects are now followed by hand, capped at five hops, with every hop
validated. The scheme is also checked *before* any coercion — prepending
`https://` first turned `file:///etc/passwd` into
`https://file:///etc/passwd`, which then parsed as scheme https and
passed. That check was unreachable code; the old test only caught it
because the mangled host happened not to resolve.

### 6. Conflicts are preserved, not resolved by convenience

When sources disagree the registry wins — but the losing value is kept and
reported. Annual figures are keyed by fiscal year so revenue in FY2025 and
FY2023 are not mistaken for a disagreement.

---

## Guarantees

| | |
|---|---|
| **Deterministic core** | Identity, management, financials and the explanation work with no API keys |
| **Wrong-company protection** | A fact is published only if its org number is confirmed at source |
| **No silent gaps** | A stage that failed is listed in `retrieval_errors`; an empty section never means "does not exist" |
| **Traceable** | Every claim carries source URL, retrieval time, `as_of`, and tier-bounded confidence |
| **Honest freshness** | Age is measured from the date the *fact* describes, not from when it was fetched |

### Known limits

- **Shareholders and beneficial owners are unavailable.** Skatteetaten's
  shareholder register is a manual CSV order; the beneficial-owner register
  requires Maskinporten credentials. The agent reports the gap rather than
  guessing.
- **Negative revenue is real.** Filed accounts can report negative total
  operating income after write-downs. These are published as filed, and
  derived ratios are suppressed when the denominator is not positive.
- **The Anthropic path is still mock-tested only.** The NVIDIA path has
  been verified live end-to-end; Anthropic has not, because no key was
  available. Treat the first Anthropic run as a shakedown.
- **Personal-detail filtering is lexical, not semantic.** Biography is
  refused by pattern match. It catches stated family, birth, death and
  hobby facts; it will not catch a paragraph of someone else's story
  written without those words.
- **Facts concentrate in `business_summary`.** Fields like
  `recent_developments` appear only when a page supports them; expect the
  stable descriptive material to dominate.
- **Proff may not be scraped.** Their terms forbid systematic automated
  retrieval. It is used only as a discovery signal, never as a fact source.

---

## Layout

```
src/
  agent.py         orchestrator; registry spine + optional enrichment
  orgnumber.py     MOD-11 validation
  brreg.py         Enhetsregisteret client (~35 fields)
  roles.py         management, board, auditor
  financials.py    filed annual accounts, KONSERN-preferring
  sources.py       7-tier reliability model, structural URL classification
  identity.py      wrong-company guard (pure, no I/O)
  discovery.py     search → verify → rank
  crawl.py         bounded same-domain follow-up planner
  extract.py       schema-guided extraction + quote verification
  explain.py       narrative + deterministic fallback
  llm.py           optional Claude client (lazy import, graceful absence)
  llm_openai.py    OpenAI-compatible backend for NVIDIA NIM and others
  page.py          page fetching, SSRF guard, HTML → text + links (stdlib only)
  conflicts.py     disagreement detection and reconciliation
  freshness.py     as_of / reporting-period semantics
  evidence.py      Evidence and FactRecord models
  http.py          retrying fetch, streamed size-capped bodies, safe redirects
  netguard.py      shared URL safety guard (private addresses, schemes, redirects)
  config.py        .env discovery that works from an installed location
  cache.py         TTL disk cache with honest stale handling
  ratelimit.py     per-process request pacing
  text.py          Norwegian-aware text folding
  batch.py         batch runner over a list of org numbers
  batch_cli.py     batch argument parsing (`signalpost-batch`)
  tinyfish_search.py  search provider adapter
  request_tracker.py  outbound request budget
  __main__.py      single-company CLI (`signalpost`)
run_batch.py       shim preserving `python run_batch.py`
tests/             440 tests, no network required
tools/              record_fixture.py regenerates the end-to-end recording
```

`pyproject.toml` is the single source of truth for metadata,
dependencies and tool configuration. `pip install .` installs two console
scripts — `signalpost` and `signalpost-batch` — so the agent runs from
anywhere without being invoked from the repository root.

`requirements.txt` pins the runtime for CI. It exists because CI should
test one known dependency set, not whatever a resolver picks today.

### Batch resume

Results are written **after every company**, through a temporary file and
an atomic rename, rather than once at the end. A fifty-company enriched
run is minutes of billable calls, and anything that ended it early — an
exception, a closed laptop, a rate limit — used to discard every profile
already finished.

`--resume` skips companies a previous run completed. Companies that
errored are retried, because a transient network failure should not
become permanent, and a retried company *replaces* its earlier record
rather than appearing twice.

Resume refuses to reuse results written under different settings:

```
$ python run_batch.py --resume                 # registry-only output exists
  resuming: 10 already completed, skipping them
  requests        0/2000

$ python run_batch.py --resume --extract
  resuming: no matching previous run (different settings or no state
  file), starting fresh
```

That refusal is the point. A registry-only profile is a valid answer to
a different question than an enriched one, and silently blending the two
into one output file is worse than redoing paid work. Progress lives in
`company_profiles.json.state.json`, so the results file stays a plain
list for anything consuming it.

Tests run on the standard library alone — `python -m unittest discover -s
tests`. There is no development dependency to install. `pytest` also runs
them unchanged if you prefer it.

CI runs `ruff` as a gate and `mypy` as a report. mypy is advisory on
purpose: the project shipped hundreds of tests and no type checking at all, and
its first pass found nine issues — one of them a live bug, where the
crawl gate compared an optional domain against a string and silently
disabled the crawl for any company without a registered website. It now
passes on all 28 source files. Making it a gate is a deliberate,
incremental change rather than something to flip on in the same commit
that introduces it.

## Configuration

| Variable | Required | Purpose |
|---|---|---|
| `TINYFISH_API_KEY` | for `--web` | Search discovery |
| `ANTHROPIC_API_KEY` | for `--extract` | Claude, via the official SDK |
| `NVIDIA_API_KEY` | for `--extract` | NVIDIA NIM or any Chat Completions endpoint |
| `LLM_PROVIDER` | no | `anthropic` \| `nvidia` \| `none`; overrides auto-detection |
| `NVIDIA_MODEL` | no | Default `nvidia/nemotron-3-super-120b-a12b` |
| `NVIDIA_BASE_URL` | no | Default `https://integrate.api.nvidia.com/v1` |
| `NVIDIA_THINKING` | no | `true` to leave the model's reasoning on |
| `NVIDIA_RPM` | no | Request pacing, default 30. `0` disables |

Extraction depth, same defaults in both entry points and settable in
both: **6 sources, 10 pages and 5 crawl links** per company. Sources stop
at six because past that the marginal results are directories repeating
Enhetsregisteret; pages exceed sources because a source costs a search
while a page costs a model call. `--max-crawl-links 0` disables the crawl
entirely, leaving homepage-only behaviour.

These bound cost, never trust — the identity gate runs first, so widening
them widens what is *considered*, not what is *accepted*.

`ANTHROPIC_AUTH_TOKEN` and `CLAUDE_CODE_OAUTH_TOKEN` are also honoured.

### Two providers, one guarantee

Claude constrains generation with `output_config.format`, so the payload
is schema-valid by construction. NVIDIA's hosted endpoint makes no such
guarantee — `guided_json` is documented only for self-hosted NIM, and
schema support is known to vary by model.

So the NVIDIA backend asks for constrained output, falls back to
instructing the schema in the prompt, and then parses defensively:
markdown fences are stripped, prose around the object is ignored, and
an unterminated object is reported as truncation rather than quietly
repaired.

**That difference cannot reach the facts.** Quoted evidence is checked
against the page by `src/extract.py`, which is the agent's own code and
behaves identically whichever provider answered. `tests/test_extraction_nvidia.py`
proves it end to end: a hallucinated quote is discarded on the NVIDIA
path exactly as it is on the Claude one.

## Dependencies

Three at runtime — `requests`, `pydantic`, `python-dotenv`. Every fact comes
from a registry record, so no HTML parser is needed; `src/page.py` flattens
markup with the standard library alone.

## Design decisions worth knowing

**Why the registry is primary.** The brief emphasised evidence over
volume, and rule 6 says "prefer first-party sources". Brreg *is* the
first-party source, and it is richer than anything a search index returns.
Building registry-first instead of search-first moved coverage from ~4
scraped facts per company to ~20 authoritative ones.

**Why freshness ignores retrieval time.** A revenue figure fetched one
minute ago may describe a fiscal year three years closed. Measuring
freshness from retrieval time marks everything fresh forever.

**Why identity resolution never involves an LLM.** It is the one judgement
the whole profile rests on, and it is the one a model will quietly get
wrong. It is 300 lines of deterministic rules instead.

**Why `facts` is separate from `evidence`.** `evidence` is the full
provenance trail for registry fields. `facts` holds claims from enrichment
that do not map onto a registry field, so the two never duplicate.