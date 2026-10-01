import json
from pathlib import Path

from src.agent import build_profile


def process_companies(
    company_numbers: list[str],
    output_file: str = "data/company_profiles.json",
) -> list[dict]:

    results = []

    for company_number in company_numbers:
        company_number = company_number.strip()

        if not company_number:
            continue

        try:
            profile = build_profile(company_number)
            results.append(profile.model_dump())

        except Exception as exc:
            results.append(
                {
                    "company_number": company_number,
                    "error": str(exc),
                }
            )

    output_path = Path(output_file)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return results


def load_company_numbers(
    input_file: str = "data/company_numbers.txt",
) -> list[str]:

    path = Path(input_file)

    return [
        line.strip()
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]