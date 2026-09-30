from datetime import datetime, timezone


def calculate_age_days(retrieved_at: str) -> float:
    retrieved = datetime.fromisoformat(retrieved_at)

    if retrieved.tzinfo is None:
        retrieved = retrieved.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)

    return max(0.0, (now - retrieved).total_seconds() / 86400)


def is_fresh(retrieved_at: str, max_age_days: int = 30) -> bool:
    return calculate_age_days(retrieved_at) <= max_age_days


def freshness_status(
    retrieved_at: str,
    max_age_days: int = 30,
) -> dict:
    age_days = calculate_age_days(retrieved_at)

    return {
        "retrieved_at": retrieved_at,
        "age_days": round(age_days, 2),
        "max_age_days": max_age_days,
        "is_fresh": age_days <= max_age_days,
    }