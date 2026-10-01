from src.batch import load_company_numbers, process_companies
from src.request_tracker import request_tracker


def main():
    company_numbers = load_company_numbers()

    print(
        f"Processing {len(company_numbers)} companies..."
    )

    results = process_companies(company_numbers)

    successful = sum(
        1
        for result in results
        if "error" not in result
    )

    failed = len(results) - successful

    print(f"Completed: {successful}")
    print(f"Failed: {failed}")

    print(
        f"Requests used: "
        f"{request_tracker.count}"
    )

    print(
        f"Requests remaining: "
        f"{request_tracker.remaining()}"
    )

    print(
        "Output: data/company_profiles.json"
    )


if __name__ == "__main__":
    main()