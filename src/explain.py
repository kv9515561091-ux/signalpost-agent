"""
Explanation.

The specification asks the agent to explain four things: what was found,
where it came from, why the source is relevant, and how confident it is.

That can be produced from the structured record alone, so it does not
depend on a language model. An LLM rewording improves fluency but is
never the only path to an explanation — losing the API must not silently
cost the profile its provenance section.

The LLM is given the verified facts and told to narrate them. It is not
asked to look anything up, and its output is checked for organisation
numbers that do not belong to the target before it is accepted.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from src import llm

_ORG_NUMBER = re.compile(r"(?<!\d)\d{3}[\s-]?\d{3}[\s-]?(\d{3})(?!\d)")


@dataclass
class Explanation:
    """A narrative plus how it was produced."""

    text: str
    generated_by: str
    llm_used: bool = False
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "generated_by": self.generated_by,
            "llm_used": self.llm_used,
            "note": self.note,
        }


def _money(amount: float | None, currency: str | None) -> str:
    if amount is None:
        return "not reported"
    unit = currency or "NOK"
    if abs(amount) >= 1_000_000_000:
        return f"{unit} {amount / 1_000_000_000:,.2f} bn"
    if abs(amount) >= 1_000_000:
        return f"{unit} {amount / 1_000_000:,.1f} m"
    return f"{unit} {amount:,.0f}"


def build_deterministic(profile: Any) -> str:
    """Compose an explanation purely from the verified record.

    This is the floor: every statement in it is traceable to a registry
    field or an evidence record, so it needs no model and cannot be wrong
    in a way the rest of the profile is not.
    """
    name = profile.company_name or "This entity"
    lines: list[str] = []

    lines.append(
        f"{name} is registered in the Norwegian Business Register "
        f"(Brønnøysundregistrene) under organization number "
        f"{profile.company_number}."
    )

    identity: list[str] = []
    if profile.legal_form:
        identity.append(profile.legal_form)
    if profile.registration_date:
        identity.append(f"registered {profile.registration_date}")
    if profile.founding_date:
        identity.append(f"established {profile.founding_date}")
    if identity:
        lines.append("It is " + ", ".join(identity) + ".")

    if profile.business_description:
        lines.append(f"Its registered purpose is: {profile.business_description}")

    if profile.industry:
        suffix = f" ({profile.industry_code})" if profile.industry_code else ""
        lines.append(f"Classified under NACE {profile.industry}{suffix}.")

    if profile.address:
        lines.append(f"The registered business address is {profile.address}.")

    if profile.employee_count is not None:
        as_of = (
            f" (as registered {profile.employee_count_as_of})"
            if profile.employee_count_as_of
            else ""
        )
        lines.append(f"It has {profile.employee_count} registered employees{as_of}.")

    # ---- Management ------------------------------------------------
    management: list[str] = []
    if profile.ceo_name:
        management.append(f"chief executive {profile.ceo_name}")
    elif profile.owner_name:
        management.append(
            f"no separate managing director; the sole owner is "
            f"{profile.owner_name}"
        )
    if profile.board_chair_name:
        management.append(f"board chaired by {profile.board_chair_name}")
    if profile.board_size:
        management.append(f"{profile.board_size} board members")
    if profile.auditor_name:
        management.append(f"auditor {profile.auditor_name}")
    if management:
        lines.append("Management: " + "; ".join(management) + ".")

    # ---- Financials ------------------------------------------------
    financials = profile.financials or []
    if financials:
        latest = financials[0]
        period = latest.fiscal_year or "the latest reported year"
        currency = profile.financial_currency or latest.currency or "NOK"
        scope = "consolidated" if latest.statement_scope == "KONSERN" else "standalone"
        lines.append(
            f"Filed accounts for {period} ({scope}, {currency}) report revenue "
            f"{_money(latest.revenue, currency)}, net result "
            f"{_money(latest.net_result, currency)} and total assets "
            f"{_money(latest.total_assets, currency)}."
        )
        if len(financials) > 1:
            lines.append(
                f"{len(financials)} consecutive years are on file, from "
                f"{financials[-1].fiscal_year} to {financials[0].fiscal_year}."
            )
    else:
        lines.append(
            "No annual accounts are on file with the registry for this "
            "entity, so no financial figures can be reported."
        )

    # ---- Status and risk -------------------------------------------
    flags: list[str] = []
    if profile.bankruptcy:
        flags.append("the entity is recorded as bankrupt")
    if profile.under_liquidation:
        flags.append("the entity is under liquidation")
    if flags:
        lines.append("Note: " + "; ".join(flags) + ".")
    elif profile.status == "aktiv":
        lines.append("The entity is recorded as active.")

    # ---- Provenance ------------------------------------------------
    lines.append("")
    lines.append(
        f"Every figure above comes from official Norwegian public records: "
        f"{len(profile.sources)} source(s) were consulted, and "
        f"{len(profile.evidence)} individual claims are recorded with their "
        f"source URL, retrieval timestamp and confidence."
    )

    if profile.retrieval_errors:
        lines.append(
            "Some sources could not be retrieved in full: "
            + "; ".join(profile.retrieval_errors)
            + ". Absence of a figure should not be read as absence of the fact."
        )

    if profile.conflicts:
        lines.append(
            f"{len(profile.conflicts)} field(s) are reported differently by "
            "different sources. The registry value is published and the "
            "disagreement is recorded rather than hidden."
        )

    if profile.facts:
        lines.append(
            f"{len(profile.facts)} additional fact(s) were extracted from "
            "verified public pages, each carrying the quoted text it came from."
        )

    return "\n".join(lines)


SYSTEM_PROMPT = """\
You write a short, factual briefing about a company from verified data.

You are given the company record and a list of sourced facts. Write 2-4
sentences of plain prose describing what the company is and what is
known about it.

Rules:
- Use only the supplied facts. Do not add anything from your own
  knowledge, and do not speculate.
- Never state a figure that is not in the supplied facts.
- Never name an organization number other than the one supplied.
- Write in plain English. Do not use markdown, bullet points or headers.
- If the facts are sparse, say briefly what is known rather than padding.
"""


def _mentions_foreign_org_number(text: str, company_number: str) -> bool:
    """Whether ``text`` cites an organisation number other than the target."""
    for match in _ORG_NUMBER.finditer(text):
        cited = match.group(0).replace(" ", "").replace("-", "")
        if cited != company_number:
            return True
    return False


def explain(profile: Any, use_llm: bool = True) -> Explanation:
    """Produce an explanation, preferring the LLM and falling back cleanly."""
    deterministic = build_deterministic(profile)

    if not use_llm or not llm.is_available():
        return Explanation(
            text=deterministic,
            generated_by="deterministic",
            llm_used=False,
            note=llm.unavailable_reason() if use_llm else "LLM disabled",
        )

    facts = [
        {
            "field": f.field,
            "value": f.value,
            "source": f.source_type,
            "confidence": f.confidence,
            "evidence": f.evidence_text[:300],
        }
        for f in profile.facts
    ]

    payload = {
        "company_name": profile.company_name,
        "company_number": profile.company_number,
        "legal_form": profile.legal_form,
        "industry": profile.industry,
        "business_description": profile.business_description,
        "address": profile.address,
        "employee_count": profile.employee_count,
        "ceo_name": profile.ceo_name,
        "owner_name": profile.owner_name,
        "financials": [
            {
                "fiscal_year": y.fiscal_year,
                "currency": y.currency,
                "revenue": y.revenue,
                "net_result": y.net_result,
                "total_assets": y.total_assets,
            }
            for y in (profile.financials or [])
        ],
        "extracted_facts": facts,
    }

    try:
        narrative = llm.complete_text(
            system=SYSTEM_PROMPT,
            user=(
                f"Verified record:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
                "\n\nWrite the briefing."
            ),
            max_tokens=1500,
        )
    except (llm.LLMUnavailable, llm.LLMRefused, llm.LLMTruncated,
            llm.LLMCallFailed) as exc:
        return Explanation(
            text=deterministic,
            generated_by="deterministic",
            llm_used=False,
            note=f"LLM narrative unavailable ({exc}); using record summary",
        )

    # The model narrating a verified record must not introduce a number
    # that is not in it.
    if _mentions_foreign_org_number(narrative, profile.company_number):
        return Explanation(
            text=deterministic,
            generated_by="deterministic",
            llm_used=False,
            note="LLM narrative cited another organization; rejected",
        )

    return Explanation(
        text=f"{narrative.strip()}\n\n{deterministic}",
        generated_by="claude",
        llm_used=True,
        note="Narrative from Claude; provenance summary from the record.",
    )