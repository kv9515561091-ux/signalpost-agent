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

python -m src.agent 989061593                      # registry spine only
python -m src.agent 989061593 --narrate            # + written explanation
python -m src.agent 989061593 --web                # + verified public sources
python -m src.agent 989061593 --extract --narrate  # everything
python -m src.agent 989061593 --json -v            # raw profile + provenance
python -m src.agent --clear-cache                  # drop cached responses

python -m unittest discover -s tests               # 188 tests
python run_batch.py                                # batch over data/company_numbers.txt
```

Optional LLM features need one extra install:

```bash
pip install -r requirements-llm.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

Without it the agent runs unchanged — see [Optional LLM](#optional-llm).

### Caching

Registry responses are cached under `data/cache/` (gitignored) for six
hours, so repeat runs cost nothing:

```
COLD: Cache: 0 hit(s), 3 written | Requests used: 3/2000
WARM: Cache: 3 hit(s), 0 written | Requests used: 0/2000
```

A cache entry past its time-to-live is **not** silently reused. It is
served only when revalidation fails, and the profile then carries
`served_stale: true` plus a retrieval gap saying so. A 404 is never
replaced by a stale body — "no such record" is the current truth.

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

### 5. Conflicts are preserved, not resolved by convenience

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
- **The LLM path is unverified against the live API.** No key was available
  during development. It is covered by 15 tests over request shape,
  refusal/truncation handling and every degradation path, but treat the
  first real run as a shakedown.
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
  extract.py       schema-guided extraction + quote verification
  explain.py       narrative + deterministic fallback
  llm.py           optional Claude client (lazy import, graceful absence)
  page.py          page fetching, SSRF guard, HTML → text (stdlib only)
  conflicts.py     disagreement detection and reconciliation
  freshness.py     as_of / reporting-period semantics
  evidence.py      Evidence and FactRecord models
  http.py          retrying fetch, streamed size-capped bodies
  cache.py         TTL disk cache with honest stale handling
  text.py          Norwegian-aware text folding
  batch.py         batch runner over a list of org numbers
  tinyfish_search.py  search provider adapter
  request_tracker.py  outbound request budget
  __main__.py      CLI
tests/             188 tests, no network required
```

Tests run on the standard library alone — `python -m unittest discover -s
tests`. There is no development dependency to install. `pytest` also runs
them unchanged if you prefer it.

## Configuration

| Variable | Required | Purpose |
|---|---|---|
| `TINYFISH_API_KEY` | for `--web` | Search discovery |
| `ANTHROPIC_API_KEY` | for `--extract` / LLM narrative | Fact extraction and explanation |

`ANTHROPIC_AUTH_TOKEN` and `CLAUDE_CODE_OAUTH_TOKEN` are also honoured.

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