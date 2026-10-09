"""Parsing and comparison of the numbers mentioned in atoms.

Numbers show up in atomic facts in many guises -- digits, plain-language words,
fractions, percentages and scale words like "million". ``parse_number`` turns
such text into plain floats so that two atoms can be compared on the numbers
they mention regardless of how those numbers happen to be written.

Spelled-out numbers are handled by ``text_to_num``, which supports several
languages; we default it to English but fall back to it for the generic cases.
"""

import math
import re
from decimal import Decimal

from text_to_num import alpha2digit

from api.models import Point

NUMBER_OVERLAP_THRESHOLD: float = 1.0

# Words that name the top of a fraction. ``text_to_num`` handles these
# inconsistently ("two thirds" -> "2 3rds", "three quarters" -> "3 quarters"),
# so fractions are resolved here, before ``alpha2digit`` runs.
_FRACTION_NUMERATORS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}
_FRACTION_DENOMINATORS = {
    "half": 2,
    "halves": 2,
    "halve": 2,
    "halved": 2,
    "halving": 2,
    "third": 3,
    "thirds": 3,
    "quarter": 4,
    "quarters": 4,
    "fourth": 4,
    "fourths": 4,
    "fifth": 5,
    "fifths": 5,
    "sixth": 6,
    "sixths": 6,
    "seventh": 7,
    "sevenths": 7,
    "eighth": 8,
    "eighths": 8,
    "ninth": 9,
    "ninths": 9,
    "tenth": 10,
    "tenths": 10,
}
# An optional numerator (word or digit) followed by a denominator word, e.g.
# "two thirds", "three-quarters", "half". A bare denominator means one of them.
_WORD_FRACTION_PATTERN = re.compile(
    r"\b(?:(?P<num>\d+|" + "|".join(_FRACTION_NUMERATORS) + r")[\s-]+)?"
    r"(?P<den>" + "|".join(_FRACTION_DENOMINATORS) + r")\b"
)
# A numeric fraction like "1/2" or "3 / 4", but not a component of a date such
# as "31/05/2026" (guarded by rejecting an adjacent digit or slash).
_SLASH_FRACTION_PATTERN = re.compile(r"(?<![\d/])(\d+)\s*/\s*(\d+)(?![\d/])")

# Number words ``text_to_num`` deliberately skips because they double as
# articles/pronouns, but which are unambiguous inside a number entity.
_EXTRA_WORD_NUMBERS = {"one": "1"}

# ``alpha2digit`` renders a scale word after a digit as a standalone integer
# rather than multiplying ("7.7 million" -> "7.7 1000000"), so we collapse those
# pairs ourselves. Ordered largest first purely for readability.
_SCALE_VALUES = (1_000_000_000_000, 1_000_000_000, 1_000_000, 1_000, 100)

# A run of digits, optionally with thousands separators and a decimal part.
_NUMBER_TOKEN = r"\d[\d,]*(?:\.\d+)?"
# The optional connector bridges phrases like "half a million" and "quarter of
# a million": once the fraction becomes a digit the leftover "(of) a/the" would
# otherwise block the scale from combining. An article is always required after
# "of" so genuine counts like "3 of 1000" are left as two separate numbers.
_SCALE_PATTERN = re.compile(
    rf"(?P<value>{_NUMBER_TOKEN})\s+(?:(?:of\s+)?(?:an?|the)\s+)?"
    rf"(?P<scale>{'|'.join(str(scale) for scale in _SCALE_VALUES)})\b"
)
_NUMBER_PATTERN = re.compile(_NUMBER_TOKEN)

# A number written as a percentage ("50%", "50 percent"), which we express as a
# fraction of one so that "50%" and "half" compare equal. "percent" needs a word
# boundary, which "percentage points" lacks, so those are left as a point count.
_PERCENT_PATTERN = re.compile(rf"(?P<value>{_NUMBER_TOKEN})\s*(?:%|per\s?cent\b)")


def _format_number(value: float) -> str:
    """Format a float as a plain decimal string, never scientific notation.

    Parsed numbers are substituted back into the text and re-scanned by
    ``_NUMBER_PATTERN``, which only understands plain decimals. ``repr`` emits
    exponents for very large or very small values (e.g. ``1e-06``, ``1e+17``),
    which the pattern would miss, so we go via ``Decimal`` to keep the shortest
    round-trip digits while forcing fixed-point form.
    """
    return format(Decimal(repr(value)), "f")


def _replace_word_fraction(match: "re.Match[str]") -> str:
    num_text = match.group("num")
    if num_text is None:
        numerator = 1
    elif num_text.isdigit():
        numerator = int(num_text)
    else:
        numerator = _FRACTION_NUMERATORS[num_text]
    denominator = _FRACTION_DENOMINATORS[match.group("den")]
    return _format_number(numerator / denominator)


def _replace_slash_fraction(match: "re.Match[str]") -> str:
    return _format_number(int(match.group(1)) / int(match.group(2)))


def _replace_percentage(match: "re.Match[str]") -> str:
    return _format_number(float(match.group("value").replace(",", "")) / 100)


def _combine_scales(text: str) -> str:
    """Collapse "<number> <scale>" pairs (e.g. "7.7 1000000") into a product."""

    def multiply(match: "re.Match[str]") -> str:
        value = float(match.group("value").replace(",", ""))
        scale = int(match.group("scale"))
        return _format_number(value * scale)

    previous = None
    while previous != text:
        previous = text
        text = _SCALE_PATTERN.sub(multiply, text)
    return text


def parse_number(number_string: str) -> list[float]:
    """
    Parse every number contained in an entity's text and return them as floats.

    Handles plain-language numbers ("three", "twenty-five", "one"), fractions
    ("two thirds", "1/2"), scale words ("2.5 million", "124 thousand"),
    decimals, percentages and thousands separators. A string may contain more
    than one number (e.g. a range like "1481 and 1482"), so a list is returned;
    a string with no number gives ``[]``.
    """
    text = number_string.lower()
    text = _SLASH_FRACTION_PATTERN.sub(_replace_slash_fraction, text)
    text = _WORD_FRACTION_PATTERN.sub(_replace_word_fraction, text)
    text = alpha2digit(text, "en")
    for word, digit in _EXTRA_WORD_NUMBERS.items():
        text = re.sub(rf"\b{word}\b", digit, text)
    text = _combine_scales(text)
    text = _PERCENT_PATTERN.sub(_replace_percentage, text)
    return [float(match.replace(",", "")) for match in _NUMBER_PATTERN.findall(text)]


def _contains_number(target: float, candidates: list[float]) -> bool:
    return any(math.isclose(target, c, abs_tol=1e-9) for c in candidates)


def get_number_overlap(new_text: str, og_text: str) -> float:
    """
    Calculates the proportion of numbers in `og_text` that are contained
    within `new_text`.
    1.0 would mean all of the original numbers are contained.
    0.0 would mean none of the original numbers are contained.

    Numbers are parsed from the full atom text (not just NER number entities)
    so that years and other numbers the tagger might skip are still compared;
    see `dev/compare_number_source.py` for the measurement behind this choice.
    Numbers are compared by value (via `parse_number`), so "7.7 million" and
    "7,700,000" count as the same number.
    """
    og_numbers = parse_number(og_text)
    new_numbers = parse_number(new_text)

    # No numbers to disagree on, so treat as a full overlap.
    if not og_numbers:
        return 1.0

    matches = [_contains_number(number, new_numbers) for number in og_numbers]
    return sum(matches) / len(og_numbers)


def check_number_match(new_atom: Point, og_atom: Point) -> bool:
    """
    Checks whether a new atom contains the same numbers as an original atom,
    above the specified threshold `NUMBER_OVERLAP_THRESHOLD`.
    """
    number_overlap = get_number_overlap(new_atom.atomic_fact, og_atom.atomic_fact)
    return number_overlap >= NUMBER_OVERLAP_THRESHOLD
