"""
Text normalisation for Norwegian company data.

Norwegian orthography resists naive ASCII folding. ``ø`` and ``æ`` are
distinct letters rather than accented forms of ``o`` and ``a``, so
Unicode decomposition leaves them untouched and a comparison against
hand-typed ASCII ("Gjorva" for "Gjørva") fails. ``å`` behaves the
opposite way, decomposing into ``a`` plus a combining ring.

One canonical folding, used by both identity matching and conflict
detection, keeps those two subsystems from disagreeing about whether two
spellings are the same value.
"""

from __future__ import annotations

import unicodedata

# Letters that survive Unicode decomposition but need explicit mapping.
_NORDIC_LETTERS = {
    "ø": "o",
    "æ": "ae",
}


def fold(text: str | None) -> str:
    """Lowercase and strip diacritics, leaving only ASCII alphanumerics.

    >>> fold("Gjørva AS")
    'gjorva as'
    >>> fold("Blæder Gudbrandsen")
    'blaeder gudbrandsen'
    """
    if not text:
        return ""

    lowered = str(text).strip().lower()

    decomposed = unicodedata.normalize("NFKD", lowered)
    without_marks = "".join(
        ch for ch in decomposed if not unicodedata.combining(ch)
    )

    for source, replacement in _NORDIC_LETTERS.items():
        without_marks = without_marks.replace(source.lower(), replacement)

    return " ".join("".join(
        ch if ch.isalnum() else " " for ch in without_marks
    ).split())


def collapse(text: str) -> str:
    """Fold, then strip whitespace entirely.

    Used where spacing is an artefact of the source rather than part of
    the value.
    """
    return fold(text).replace(" ", "")
