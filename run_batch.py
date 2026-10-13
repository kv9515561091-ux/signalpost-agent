"""CLI entry point: ``python run_batch.py``."""

from src.batch import load_company_numbers, process_companies, summarise
from src.request_tracker import request_tracker


def main() -> None:
    company_numbers = load_company_numbers()

    if not company_numbers:
        print("No company numbers found in data/company_numbers.txt")
        return

    print(f"Building {len(company_numbers)} company profiles...")

    results = process_companies(company_numbers)
    stats = summarise(results)

    for key, value in stats.items():
        print(f"  {key:<16} {value}")

    tracker = request_tracker.summary()
    print(
        f"  requests        {tracker['requests_used']}"
        f"/{tracker['request_limit']}"
    )
    print("  output          data/company_profiles.json")


if __name__ == "__main__":
    main()
