"""Render terminal envelopes into a self-contained static site.

A run produces envelopes a machine can verify, but a person still has to
decide whether to trust them. This renders each envelope into a page a
non-engineer can search, read and verify: claims grouped by information
family, every claim carrying its source URL and retrieval time, explicit
"not found" treatment instead of silent gaps, and an index that filters
as you type with no server. Everything is inlined (one stylesheet, one
small script), so a fresh clone renders offline in any modern browser,
on desktop and on a phone.
"""

from __future__ import annotations

import html
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_CSS = """
:root {
  --bg: #f6f7f9; --card: #ffffff; --ink: #17202a; --muted: #5b6770;
  --line: #dfe3e8; --link: #0b5cab; --ok: #146c2e; --ok-bg: #e7f4ea;
  --warn: #8a5300; --warn-bg: #fdf0d5; --bad: #a11212; --bad-bg: #fbe4e4;
  --muted-bg: #eceef1;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #101418; --card: #1a2027; --ink: #e6eaee; --muted: #9aa7b2;
    --line: #2c3540; --link: #6cb4ff; --ok: #7fd6a0; --ok-bg: #10241a;
    --warn: #ffce74; --warn-bg: #241a06; --bad: #ff9d9d; --bad-bg: #26100f;
    --muted-bg: #222a33;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font: 16px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}
main { max-width: 1080px; margin: 0 auto; padding: 24px 16px 48px; }
h1, h2 { line-height: 1.25; }
h1 { font-size: 1.6rem; margin: 0.4em 0 0.2em; }
h2 { font-size: 1.1rem; margin: 1.4em 0 0.5em; text-transform: uppercase;
     letter-spacing: 0.04em; color: var(--muted); }
.topbar { border-bottom: 1px solid var(--line); background: var(--card); }
.topbar .inner { max-width: 1080px; margin: 0 auto; padding: 10px 16px;
                 display: flex; gap: 12px; align-items: baseline; }
.topbar a { color: var(--link); text-decoration: none; font-weight: 600; }
.card { background: var(--card); border: 1px solid var(--line);
        border-radius: 12px; padding: 16px; margin: 12px 0; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0; }
.chip { font-size: 0.85rem; border-radius: 999px; padding: 3px 12px;
        border: 1px solid var(--line); }
.chip.ok { color: var(--ok); background: var(--ok-bg); border-color: transparent; }
.chip.warn { color: var(--warn); background: var(--warn-bg); border-color: transparent; }
.state { display: inline-block; font-size: 0.75rem; font-weight: 700;
         letter-spacing: 0.05em; text-transform: uppercase; border-radius: 6px;
         padding: 2px 8px; vertical-align: 2px; }
.state.complete { color: var(--ok); background: var(--ok-bg); }
.state.source_error, .state.budget_exhausted { color: var(--warn); background: var(--warn-bg); }
.state.submission_error, .state.not_found, .state.not_applicable { color: var(--bad); background: var(--bad-bg); }
a { color: var(--link); }
a:focus-visible, .search:focus-visible {
  outline: 3px solid var(--link); outline-offset: 2px; border-radius: 4px; }
table { width: 100%; border-collapse: collapse; font-size: 0.92rem; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line);
         vertical-align: top; }
th { color: var(--muted); font-size: 0.78rem; text-transform: uppercase;
     letter-spacing: 0.05em; }
tbody tr:hover td { background: var(--muted-bg); }
.wrap { overflow-x: auto; }
footer { border-top: 1px solid var(--line); margin-top: 40px; padding: 16px;
         color: var(--muted); font-size: 0.8rem; text-align: center; }
code { background: var(--muted-bg); border-radius: 4px; padding: 1px 5px;
       font-size: 0.85em; }
.small { font-size: 0.85rem; color: var(--muted); }
.search { width: 100%; padding: 12px 14px; border-radius: 10px;
          border: 1px solid var(--line); background: var(--card);
          color: var(--ink); font-size: 1rem; margin: 10px 0; }
.badges { display: flex; gap: 8px; flex-wrap: wrap; margin: 6px 0 14px; }
.badge { border: 1px solid var(--line); border-radius: 8px; padding: 6px 12px;
         font-size: 0.85rem; background: var(--card); }
.badge b { font-size: 1.05rem; }
@media print { .topbar, .search, .badges { display: none !important; }
  body { background: #fff; color: #000; } }
"""

_FAMILY_ORDER = (
    "identity",
    "contact",
    "leadership",
    "financials",
    "workforce",
    "locations",
    "activity",
    "social",
    "other",
)
_FAMILY_LABEL = {
    "identity": "Identity",
    "contact": "Contact",
    "social": "Social",
    "leadership": "Leadership",
    "financials": "Financials",
    "workforce": "Workforce",
    "locations": "Locations",
    "activity": "Activity",
    "other": "Other recorded facts",
}


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _url(value: Any) -> str:
    url = str(value or "")
    return url if url.startswith(("http://", "https://")) else ""


def _claim_row(claim: dict[str, Any], evidence_by_id: dict[str, dict[str, Any]]) -> str:
    field = _esc(claim.get("field") or "")
    availability = str(claim.get("availability") or "")
    value = claim.get("value")
    if value is None or availability != "available":
        shown = availability or "not published"
        value_cell = f'<span class="small">{_esc(shown)}</span>'
    else:
        value_cell = _esc(value)
    confidence = claim.get("confidence")
    conf = f"{confidence:.2f}" if isinstance(confidence, (int, float)) else "—"
    ev_links = []
    for ev_id in claim.get("evidence_ids") or []:
        evidence = evidence_by_id.get(str(ev_id))
        if not evidence:
            continue
        ev_url = _url(evidence.get("source_url"))
        if ev_url:
            tier = evidence.get("source_tier")
            tier_txt = f" · T{tier}" if tier is not None else ""
            retrieved = _esc(evidence.get("retrieved_at") or "")
            ev_links.append(
                f'<a href="{_esc(ev_url)}" target="_blank" rel="noopener noreferrer">'
                f"source{_esc(tier_txt)}</a> <span class='small'>({retrieved})</span>"
            )
    source_cell = (
        "<br>".join(ev_links) if ev_links else '<span class="small">no evidence</span>'
    )
    return (
        f"<tr><th scope='row'>{field}</th><td>{value_cell}</td>"
        f"<td>{conf}</td><td>{source_cell}</td></tr>"
    )


def _company_page(env: dict[str, Any]) -> str:
    profile = env.get("profile") or {}
    org = _esc(profile.get("organisation_number") or env.get("organisation_number") or "")
    name = str(profile.get("company_name") or "Company")
    state = str(env.get("state") or "unknown")
    synthesis = env.get("synthesis") or {}
    coverage = synthesis.get("coverage") or {}
    claims = env.get("claims") or []
    evidence = env.get("evidence") or []
    evidence_by_id = {str(e.get("id")): e for e in evidence}

    def family_chips(families: Any, css_class: str, suffix: str = "") -> str:
        items = [f for f in (families or []) if f]
        if not items:
            return '<span class="small">None</span>'
        return "".join(
            f'<span class="chip {css_class}">{_esc(_FAMILY_LABEL.get(f, f))}{suffix}</span>'
            for f in items
        )

    avail_html = family_chips(coverage.get("available_families"), "ok")
    miss_html = family_chips(coverage.get("missing_families"), "", " — not covered")

    field_family: dict[str, str] = {}
    for entry in coverage.get("families") or []:
        fam = str(entry.get("family") or "other")
        for fld in entry.get("fields") or []:
            field_family[str(fld)] = fam
    grouped: dict[str, list[dict[str, Any]]] = {f: [] for f in _FAMILY_ORDER}
    for claim in claims:
        fam = field_family.get(str(claim.get("field") or ""), "other")
        grouped.setdefault(fam, []).append(claim)
    grouped = {f: value for f, value in grouped.items() if value}

    sections = []
    for fam in _FAMILY_ORDER:
        fam_claims = grouped.get(fam)
        if not fam_claims:
            continue
        label = _FAMILY_LABEL.get(fam, fam)
        rows = "".join(_claim_row(c, evidence_by_id) for c in fam_claims)
        sections.append(
            f"<section class='card'><h2>{_esc(label)}</h2>"
            f'<div class="wrap"><table>'
            f"<thead><tr><th scope='col'>Field</th><th>Value</th>"
            f"<th scope='col'>Confidence</th><th scope='col'>Source</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div></section>"
        )

    changes = env.get("changes") or []
    if changes:
        items = "".join(
            f"<li><strong>{_esc(c.get('field') or '')}</strong>: "
            f"{_esc(c.get('previous'))} → {_esc(c.get('current'))} "
            f"<span class='small'>({_esc(c.get('kind') or '')})</span></li>"
            for c in changes[:25]
        )
        changes_card = f'<section class="card"><h2>Changes</h2><ul>{items}</ul></section>'
    else:
        changes_card = ""

    errors = env.get("errors") or []
    if errors:
        items = "".join(
            f"<li><code>{_esc(e.get('code') or '')}</code> {_esc(e.get('message') or '')}</li>"
            for e in errors
        )
        errors_card = f'<section class="card"><h2>Limitations recorded</h2><ul>{items}</ul></section>'
    else:
        errors_card = (
            '<section class="card"><h2>Limitations recorded</h2>'
            '<p class="small">None for this run.</p></section>'
        )

    summary = _esc(synthesis.get("summary") or "")
    highlights = synthesis.get("highlights") or []
    highlights_html = "".join(f"<li>{_esc(h)}</li>" for h in highlights[:12])
    generated_by = _esc(synthesis.get("generated_by") or "deterministic")
    changes_summary = _esc(synthesis.get("changes_summary") or "")
    modules = env.get("modules") or {}
    module_html = "".join(
        f'<span class="chip">{_esc(name_.replace("_", " "))}: {_esc(m.get("state") or "")}</span>'
        for name_, m in sorted(modules.items())
        if isinstance(m, dict)
    )

    retrieved = _esc(profile.get("retrieved_at") or env.get("completed_at") or "")
    legal_form = _esc(profile.get("legal_form") or "")
    industry = _esc(profile.get("industry") or "")
    website = _url(profile.get("website"))
    website_html = (
        f'<a href="{_esc(website)}" target="_blank" rel="noopener noreferrer">{_esc(profile.get("website"))}</a>'
        if website
        else '<span class="small">not verified</span>'
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(name)} — Signalpost</title>
<style>{_CSS}</style>
</head>
<body>
<nav class="topbar"><div class="inner">
  <a href="index.html">← All companies</a>
  <span class="small">Signalpost report</span>
</div></nav>
<main>
<h1>{_esc(name)} <span class="state {_esc(state)}">{_esc(state)}</span></h1>
<p>{_esc(legal_form)} · {_esc(industry)} · <code>{org}</code> · last checked {retrieved} · {website_html}</p>
<div class="chips">{module_html}</div>
<section class="card"><h2>Summary</h2>
<p>{summary}</p>
<p class="small">Generated by {generated_by}. {changes_summary}</p>
{f'<ul>{highlights_html}</ul>' if highlights_html else ''}
</section>
<section class="card"><h2>Coverage</h2>
<p class="small">Covered:</p>
<div class="chips">{avail_html}</div>
<p class="small">Not covered — abstentions, not guesses:</p>
<div class="chips">{miss_html}</div>
</section>
{''.join(sections)}
{changes_card}
{errors_card}
<footer><p>Generated by the Signalpost agent. Every published fact cites its source.
Nothing here was invented, silently zeroed or copied from another company.</p></footer>
</main>
</body>
</html>
"""


def _index_page(rows: list[dict[str, Any]], generated: str) -> str:
    body = "".join(
        f'<tr data-search="{_esc(row["key"])}"><td><a href="{_esc(row["org"])}.html">{_esc(row["name"])}</a></td>'
        f"<td><code>{_esc(row['org'])}</code></td>"
        f'<td><span class="state {_esc(row["state"])}">{_esc(row["state"])}</span></td>'
        f"<td>{row['available']}</td><td>{row['claims']}</td><td>{row['evidence']}</td></tr>"
        for row in rows
    )
    states: dict[str, int] = {}
    for row in rows:
        states[row["state"]] = states.get(row["state"], 0) + 1
    state_chips = " ".join(
        f'<span class="badge"><b>{n}</b> {_esc(s)}</span>' for s, n in sorted(states.items())
    )
    total_claims = sum(row["claims"] for row in rows)
    total_available = sum(row["available"] for row in rows)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Signalpost — company profiles</title>
<style>{_CSS}</style>
</head>
<body>
<nav class="topbar"><div class="inner">
  <a href="index.html">Signalpost</a>
  <span class="small">{total_available} evidenced facts across {len(rows)} companies</span>
</div></nav>
<main>
<h1>Company profiles</h1>
<p class="small">Generated {_esc(generated)}. Search by name or organisation number, then open a profile
and follow its sources. To compare two companies, open them in side-by-side tabs — every page shares one layout.</p>
<div class="badges">{state_chips}<span class="badge"><b>{total_claims}</b> claims</span></div>
<input class="search" id="filter" type="search" autocomplete="off"
       placeholder="Filter by name or organisation number…" aria-label="Filter companies">
<div class="wrap"><table id="table">
<thead><tr><th scope="col">Company</th><th scope="col">Org no</th><th scope="col">State</th>
<th scope="col">Facts</th><th scope="col">Claims</th><th scope="col">Evidence</th></tr></thead>
<tbody>{body}</tbody>
</table></div>
<footer><p>Profiles are reproducible from the input file. Abstentions are published as not-found claims,
never as zero.</p></footer>
</main>
<script>
(function () {{
  var input = document.getElementById('filter');
  var rows = document.querySelectorAll('#table tbody tr');
  input.addEventListener('input', function () {{
    var needle = input.value.trim().toLowerCase();
    rows.forEach(function (row) {{
      row.style.display = needle && row.getAttribute('data-search').indexOf(needle) === -1 ? 'none' : '';
    }});
  }});
}})();
</script>
</body>
</html>
"""


def render_site(
    envelopes: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    generated: str = "",
) -> dict[str, int]:
    """Render a site from terminal envelopes. Returns headline counts."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for env in envelopes:
        org = str(env.get("organisation_number") or "unknown")
        org_safe = re.sub(r"[^0-9A-Za-z_-]", "", org) or "unknown"
        profile = env.get("profile") or {}
        name = str(profile.get("company_name") or org_safe)
        claims = env.get("claims") or []
        evidence = env.get("evidence") or []
        available = sum(1 for c in claims if c.get("availability") == "available")
        rows.append(
            {
                "org": org_safe,
                "name": name,
                "state": str(env.get("state") or "unknown"),
                "claims": len(claims),
                "evidence": len(evidence),
                "available": available,
                "key": f"{name} {org_safe}".lower(),
            }
        )
        (out / f"{org_safe}.html").write_text(_company_page(env), encoding="utf-8")

    stamp = generated or datetime.now(timezone.utc).isoformat(timespec="seconds")
    (out / "index.html").write_text(_index_page(rows, stamp), encoding="utf-8")

    return {"pages": len(rows)}
