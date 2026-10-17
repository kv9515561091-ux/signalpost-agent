"""
Verify a generated corpus before it is submitted.

Checks the things that would disqualify the entry, rather than counting
what is there. A corpus that is quietly missing a field, or carrying a
fact from the wrong company, looks identical to a good one from the
outside.

    python tools/verify_corpus.py data/registry_1000.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ORG_NUMBER = re.compile(r"\b\d{9}\b")


def main() -> int:
    parser = argparse.ArgumentParser(prog="verify_corpus.py")
    parser.add_argument("corpus", nargs="+")
    parser.add_argument(
        "--min-profiles",
        type=int,
        default=1000,
        help="Fail below this many profiles.",
    )
    args = parser.parse_args()

    total_profiles = 0
    total_facts = 0
    problems: list[str] = []

    for path in args.corpus:
        target = Path(path)
        if not target.is_file():
            problems.append(f"{path}: missing")
            continue

        rows = json.loads(target.read_text(encoding="utf-8"))
        total_profiles += len(rows)

        facts = [f for r in rows for f in (r.get("facts") or [])]
        total_facts += len(facts)

        missing_evidence = [f for f in facts if not f.get("evidence_text")]
        missing_source = [f for f in facts if not f.get("source_url")]
        unverified = [f for f in facts if f.get("source_tier") == 7]

        foreign = [
            f
            for r in rows
            for f in (r.get("facts") or [])
            for number in ORG_NUMBER.findall(
                (f.get("evidence_text") or "") + " " + (f.get("source_url") or "")
            )
            if number != r["company_number"]
        ]

        duplicates = len(rows) - len({r.get("company_number") for r in rows})

        print(f"\n{target.name}")
        print(f"  profiles                {len(rows)}")
        print(f"  with a company name     {sum(1 for r in rows if r.get('company_name'))}")
        print(f"  with a role list        {sum(1 for r in rows if r.get('roles'))}")
        print(f"  with financials         {sum(1 for r in rows if r.get('financials'))}")
        print(f"  extracted facts         {len(facts)}")
        print(f"  narratives              {sum(1 for r in rows if r.get('explanation'))}")

        # Accuracy invariants. These are what disqualify an entry.
        for label, offending in (
            ("facts without evidence", missing_evidence),
            ("facts without a source URL", missing_source),
            ("facts from an unverified tier", unverified),
            ("facts citing another company", foreign),
        ):
            count = len(offending)
            mark = "ok" if count == 0 else "FAIL"
            print(f"  {label:<24} {count}  [{mark}]")
            if count:
                problems.append(f"{path}: {count} {label}")
                problems.append(f"    e.g. {offending[0]}")

        if duplicates:
            print(f"  duplicate profiles      {duplicates}  [FAIL]")
            problems.append(f"{path}: {duplicates} duplicate org numbers")
        else:
            print("  duplicate profiles      0  [ok]")

    print()
    print(f"TOTAL: {total_profiles} profiles, {total_facts} extracted facts")

    if total_profiles < args.min_profiles:
        problems.append(
            f"only {total_profiles} profiles, need {args.min_profiles}"
        )

    if problems:
        print("\nPROBLEMS")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("\nAll accuracy invariants hold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
