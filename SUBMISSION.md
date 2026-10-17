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
420b95b
```

CI is green on Python 3.10–3.13. Verify the exact hash at the moment of
submission rather than relying on this line.

## 3. One command to run it

```bash
python run_daily.py
```

That runs the full pipeline over every organization number in
`data/competition_companies.txt`: registry lookup, web discovery,
quote-verified extraction, narrative, change detection against the
previous run, and a request budget that degrades rather than fails.

Prerequisites:

```bash
pip install -r requirements.txt
```

And two keys in `.env`:

```
TINYFISH_API_KEY=...      # search discovery
NVIDIA_API_KEY=...        # extraction and narrative
```

`python run_daily.py` is resumable — re-running it after an interruption
skips companies already completed and reports only what changed.

## 4. Model and API details

| Component | Provider | Cost |
|---|---|---|
| Company identity, management, filed accounts | **Brønnøysundregistrene** (`data.brreg.no`) | Free, no key, no auth |
| Web discovery | **TinyFish** search API | Billable per call — see §5 |
| Fact extraction + narrative | **NVIDIA NIM**, `nvidia/nemotron-3-super-120b-a12b` | No per-token charge on the hosted build endpoint |
| Anthropic Claude | Supported, **not verified live** | — |

The agent speaks the Chat Completions shape directly over HTTP, so the
NVIDIA path adds no dependency. `anthropic` is an optional extra
(`pip install -e ".[anthropic]"`) and is mock-tested only; no Anthropic
key was available during development.

Extraction requests run at `temperature=0.0`, falling back to `1.0`
only if an endpoint rejects it. Model calls are paced to
`NVIDIA_RPM` (default 30, below the observed ~40 RPM ceiling) because
NVIDIA returns 429 with no `Retry-After`.

## 5. Expected run costs

Declared in calls, since search is the only billable external API.

| | Per 100 companies |
|---|---|
| Brreg requests | 300 |
| Search calls | ~200 (2 per company; cached by query) |
| Total outbound requests | **~1,600–1,750** |
| LLM calls | ~600–800 |
| NVIDIA spend | $0 (hosted build endpoint) |
| **Billable external** | **~200 TinyFish calls** |

Enrichment depth is 8 sources / 14 pages per company. Measured on a
138-company fully-enriched run:

| Metric | Value |
|---|---|
| Requests | ~20 per company |
| Facts extracted | 616 (4.5 per company) |
| Companies with at least one fact | 76 / 138 |
| Narratives produced | 126 / 138 |
| Same-domain pages crawled | 237 |
| **Facts missing evidence** | **0** |
| **Facts from an unverified (tier 7) source** | **0** |

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
  and multi-year filed accounts where they exist.
- **138** of those are additionally enriched with verified web sources,
  quote-backed extraction, same-domain crawling and a written narrative.

Registry-only is not a lesser record. The registry *is* the authoritative
source for everything it holds, and enrichment adds facts the registry
does not carry rather than correcting it. The enriched subset shows
what the enrichment layer contributes.

## 7. What the agent produces

Per company, one JSON profile containing:

- **Registry fields** — legal name and previous names, legal form,
  status, industry and code, business description, founding and
  registration dates, addresses, municipality, employee count, contact
  details, group status, bankruptcy and liquidation flags.
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
says.

**Identity is verified before extraction, not after.** Sources are
classified into seven reliability tiers; listing pages, person pages and
foreign directory IDs are refused structurally. The org number settles
identity — a name never does on its own.

**A verified first-party page may only enqueue same-domain follow-ups**,
which inherit its verification rather than earning their own. The
crawl cannot widen the trust boundary.

**Every redirect hop is re-validated.** A discovered URL that redirects
to a private address is refused mid-chain; the guard applies to the URL
the agent ends up at, not the one it started from.

**Refusals are recorded, not hidden.** Rejected claims appear in
`extraction_rejected` with the reason.

## 9. Update behaviour

The daily test re-runs the same companies, so the agent compares each
profile against the one it stored last time and reports what moved.

- Registry, contact, management, financial and extracted-fact changes
  are tagged separately, because a registry field changing is the
  company changing while a fact changing may only be a page rewritten.
- Every change carries the previous value, the new value, and the kind.
- Financial figures are keyed by fiscal year, so a newly filed year reads
  as one addition rather than making every later year look restated.
- Enhetsregisteret exposes no change signal — no `endrettsdato`, no
  ETag, no `Last-Modified` — verified against the live API, so the
  comparison is made against stored profiles rather than inferred.

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