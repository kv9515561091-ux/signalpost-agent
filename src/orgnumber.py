"""
Norwegian organization number (organisasjonsnummer) handling.

A Norwegian organization number is 9 digits. The first 8 digits are the
serial part; the 9th is a MOD-11 control digit derived from the first 8.

Checking this locally rejects typos before any outbound HTTP request is made,
which matters because every network call in this agent is billable.
"""

from __future__ import annotations

# Control-digit weights for the first 8 digits.
MOD11_WEIGHTS: tuple[int, ...] = (3, 2, 7, 6, 5, 4, 3, 2)

ORG_NUMBER_LENGTH = 9


class InvalidCompanyNumber(ValueError):
    """Raised when a string cannot be a valid Norwegian org number."""


def normalize(value: str | int) -> str:
    """Strip all non-digit characters.

    Accepts the separators Norwegian registries commonly emit
    (``"983 887 457"``, ``"983-887-457"``) as well as integers.

    >>> normalize("983 887 457")
    '983887457'
    """
    return "".join(ch for ch in str(value) if ch.isdigit())


def mod11_control_digit(serial: str) -> int | None:
    """Compute the MOD-11 control digit for the first 8 digits.

    Returns ``None`` when the serial cannot yield a valid control digit
    (a remainder that would require the digit 10).
    """
    weighted = sum(int(d) * w for d, w in zip(serial, MOD11_WEIGHTS))
    check = 11 - (weighted % 11)

    if check == 11:
        return 0
    if check == 10:
        return None
    return check


def is_valid(value: str | int, *, strict: bool = True) -> bool:
    """Return whether ``value`` is a valid Norwegian organization number.

    Args:
        value: The candidate number in any separator style.
        strict: When ``True`` (default) the MOD-11 control digit must
            check out. Set ``False`` to accept a well-formed 9-digit
            number that fails checksum, which is useful only if you are
            deliberately probing a malformed registry record.
    """
    digits = normalize(value)

    if len(digits) != ORG_NUMBER_LENGTH:
        return False
    if not strict:
        return True

    expected = mod11_control_digit(digits[:8])
    return expected is not None and expected == int(digits[8])


def validate(value: str | int) -> str:
    """Normalize and validate, returning the canonical 9-digit form.

    Raises:
        InvalidCompanyNumber: If the number is malformed or fails
            the MOD-11 checksum.
    """
    digits = normalize(value)

    if len(digits) != ORG_NUMBER_LENGTH:
        raise InvalidCompanyNumber(
            f"Organization number must be {ORG_NUMBER_LENGTH} digits, "
            f"got {len(digits)}: {value!r}"
        )

    if not is_valid(digits):
        raise InvalidCompanyNumber(
            f"Invalid Norwegian organization number (MOD-11 checksum "
            f"failed): {value!r}"
        )

    return digits
