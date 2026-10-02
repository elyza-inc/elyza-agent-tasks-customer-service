"""Shared scalar-type rules for generation and evaluation."""

from __future__ import annotations

from datetime import date
from math import isfinite
from pathlib import Path
import re
from typing import Any
import unicodedata

import yaml


SEED_SCHEMA_PATH = Path(__file__).with_name("seed_schema.yaml")
PHONE_COLUMNS = frozenset({"phone", "phone_number"})
PHONE_SEPARATOR_PATTERN = re.compile(r"[-‐‑‒–—―−ーｰ\s]")
PROVENANCE_PHONE_SEPARATOR_PATTERN = re.compile(r"[-‐‑‒–—―−ーｰ\s、,，]")
PROVENANCE_TEXT_SEPARATOR_PATTERN = re.compile(r"[\s、,，。・「」『』\"']")
DATE_PATTERN = re.compile(r"(\d{4})[-/.年](\d{1,2})[-/.月](\d{1,2})日?")
DATETIME_VALUE_PATTERN = re.compile(
    r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})[ T](\d{2}):(\d{2})"
)
TIME_VALUE_PATTERN = re.compile(r"([01]\d|2[0-3]):([0-5]\d)")
ARABIC_NUMBER_PATTERN = re.compile(
    r"(?<![\d.:\-])\d[\d,]*(?:\.\d+)?(?![\d.:\-])"
)
JAPANESE_NUMBER_KANJI = "零一二三四五六七八九"
JAPANESE_INTEGER_PATTERN = re.compile(r"[〇零一二三四五六七八九十百千万]+")
JAPANESE_NUMBER_READINGS = (
    "れい", "いち", "に", "さん", "よん", "ご", "ろく", "なな", "はち", "きゅう",
)
TIME_NUMBER_BOUNDARY = "0-9〇零一二三四五六七八九十"
OPERATIONAL_ENUM_VALUES = {
    "reissue_count": frozenset((0,)),
}
# A spoken short form ("素泊まり") names a plan by its first characters.
MIN_PLAN_NAME_PREFIX_LENGTH = 4
SPOKEN_EQUIVALENT_REPLACEMENTS = (("ジェイ", "ジェー"),)
SPOKEN_PHONE_DIGITS = (
    ("ゼロ", "0"), ("レイ", "0"), ("イチ", "1"), ("ニ", "2"), ("サン", "3"),
    ("ヨン", "4"), ("シチ", "7"), ("シ", "4"), ("ゴ", "5"), ("ロク", "6"),
    ("ナナ", "7"), ("ハチ", "8"), ("キュウ", "9"), ("ク", "9"), ("の", ""), ("ノ", ""),
)


def _load_scalar_types() -> frozenset[str]:
    """Load ``definitions.scalar_type_values`` from the object-root YAML schema.

    A missing, non-list, duplicate, or non-string value raises ``RuntimeError``.
    """

    value = yaml.safe_load(SEED_SCHEMA_PATH.read_text(encoding="utf-8"))
    raw = value.get("definitions", {}).get("scalar_type_values") if isinstance(value, dict) else None
    if not isinstance(raw, list) or not raw or any(not isinstance(item, str) for item in raw):
        raise RuntimeError(f"{SEED_SCHEMA_PATH}: definitions.scalar_type_values must be a string list")
    if len(raw) != len(set(raw)):
        raise RuntimeError(f"{SEED_SCHEMA_PATH}: definitions.scalar_type_values contains duplicates")
    return frozenset(raw)


SCALAR_TYPES = _load_scalar_types()
JSON_SCHEMA_TYPES = {
    type_name: "string" if type_name in {"date", "time", "enum"} else type_name
    for type_name in SCALAR_TYPES
}


def python_value_type(value: Any) -> str:
    """Return the scalar type name for a Python value, or its class name."""

    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    return type(value).__name__


def value_matches_type(value: Any, expected: str) -> bool:
    """Return whether a finite scalar satisfies one declared type."""

    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return (
            isinstance(value, int) and not isinstance(value, bool)
        ) or isinstance(value, float) and isfinite(value)
    if expected in {"string", "date", "time", "enum"}:
        return isinstance(value, str)
    return False


def normalize_typed_value(value: Any, expected: str) -> Any:
    """Return a validated scalar; JSON ``number`` values use float representation.

    Invalid values raise ``ValueError``. No value is rounded or truncated.
    """

    if not value_matches_type(value, expected):
        raise ValueError(f"expected {expected}, got {python_value_type(value)}")
    return float(value) if expected == "number" else value


def _normalized_match_value(value: Any, column: str, value_type: str | None) -> Any:
    """Normalize a value only for exact matching; stored values are never returned."""

    if value_type == "number" and value_matches_type(value, value_type):
        return float(value)
    if not isinstance(value, str):
        return value
    if column in PHONE_COLUMNS or column.endswith("_phone_number"):
        return PHONE_SEPARATOR_PATTERN.sub("", unicodedata.normalize("NFKC", value))
    if column == "email" or column.endswith("_email"):
        # Spoken addresses carry no letter case.
        return unicodedata.normalize("NFKC", value).strip().casefold()
    if column == "kana" or column.endswith("_kana"):
        # A reading carries no script: spoken or transcribed hiragana names the same katakana value.
        text = "".join(unicodedata.normalize("NFKC", value).split())
        return "".join(chr(ord(ch) + 0x60) if "ぁ" <= ch <= "ゖ" else ch for ch in text)
    if value_type == "date" or column == "birthdate" or column.endswith("_date"):
        return _normalized_date(value) or value.strip()
    return value.strip()


def _normalized_date(value: str) -> str | None:
    """Return ISO text for Y-M-D using ``-``, ``/``, ``.``, or Japanese separators.

    Full-width characters and one-digit months/days are accepted. Invalid or
    impossible dates return ``None``.
    """

    match = DATE_PATTERN.fullmatch(unicodedata.normalize("NFKC", value).strip())
    if match is None:
        return None
    try:
        return date(*(int(part) for part in match.groups())).isoformat()
    except ValueError:
        return None


def _japanese_number_forms(value: int) -> set[str]:
    """Return digit, kanji, hiragana, and katakana forms for 0 through 59."""

    if not 0 <= value <= 59:
        return set()
    tens, ones = divmod(value, 10)
    kanji = ""
    reading = ""
    if tens:
        kanji = ("" if tens == 1 else JAPANESE_NUMBER_KANJI[tens]) + "十"
        reading = ("" if tens == 1 else JAPANESE_NUMBER_READINGS[tens]) + "じゅう"
    if ones or not tens:
        kanji += JAPANESE_NUMBER_KANJI[ones]
        reading += JAPANESE_NUMBER_READINGS[ones]
    katakana = "".join(
        chr(ord(character) + 0x60) if "ぁ" <= character <= "ゖ" else character
        for character in reading
    )
    return {str(value), kanji, kanji.replace("零", "〇"), reading, katakana}


def _japanese_integer(value: str) -> int:
    digits = {
        character: index for index, character in enumerate(JAPANESE_NUMBER_KANJI)
    }
    digits["〇"] = 0
    total = 0
    section = 0
    digit = 0
    for character in value:
        if character in digits:
            digit = digits[character]
        elif character == "万":
            total += (section + digit or 1) * 10_000
            section = 0
            digit = 0
        else:
            unit = {"十": 10, "百": 100, "千": 1000}[character]
            section += (digit or 1) * unit
            digit = 0
    return total + section + digit


def _compact_provenance_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    for source, replacement in SPOKEN_EQUIVALENT_REPLACEMENTS:
        normalized = normalized.replace(source, replacement)
    normalized = JAPANESE_INTEGER_PATTERN.sub(
        lambda match: str(_japanese_integer(match.group(0))), normalized
    )
    return PROVENANCE_TEXT_SEPARATOR_PATTERN.sub("", normalized)


def provenance_text_contains_declared_spoken(
    candidate: Any, spoken: Any
) -> bool:
    """Return whether text contains one package-declared spoken scalar form."""

    if not isinstance(candidate, str) or not isinstance(spoken, str) or not spoken:
        return False
    return _compact_provenance_text(spoken) in _compact_provenance_text(candidate)


def _provenance_text_contains_time(candidate: str, expected: str) -> bool:
    """Match an ``HH:MM`` value in Japanese clock text, including spoken forms."""

    match = TIME_VALUE_PATTERN.fullmatch(expected.strip())
    if match is None:
        return False
    hour, minute = (int(part) for part in match.groups())
    hour_forms = _japanese_number_forms(hour)
    if hour >= 13:
        twelve_hour_forms = _japanese_number_forms(hour - 12)
        hour_forms.update(
            f"{period}{form}"
            for period in ("午後", "夜", "夜の")
            for form in twelve_hour_forms
        )
    hours = "|".join(re.escape(form) for form in sorted(hour_forms, key=len, reverse=True))
    if minute == 0:
        minutes = r"(?:(?:00|0|零|〇|れい|レイ)分)?"
    else:
        minute_forms = _japanese_number_forms(minute)
        minutes = rf"(?:{'|'.join(re.escape(form) for form in sorted(minute_forms, key=len, reverse=True))})分"
    pattern = rf"(?<![{TIME_NUMBER_BOUNDARY}])(?:{hours})時{minutes}"
    return re.search(pattern, candidate) is not None


def _provenance_text_contains_date(candidate: str, expected: str) -> bool:
    """Match a full date or its unambiguous Japanese month/day expression."""

    expected_date = _normalized_date(expected)
    if expected_date is None:
        return False
    if any(
        _normalized_date(match.group(0)) == expected_date
        for match in DATE_PATTERN.finditer(candidate)
    ):
        return True
    parsed = date.fromisoformat(expected_date)
    return any(
        f"{month}月{day}日" in candidate
        for month in _japanese_number_forms(parsed.month)
        for day in _japanese_number_forms(parsed.day)
    )


def _provenance_text_contains_datetime(candidate: str, expected: str) -> bool:
    """Match an ISO date-time against full or Japanese date/time text."""

    match = DATETIME_VALUE_PATTERN.fullmatch(expected.strip())
    if match is None:
        return False
    year, month, day, hour, minute = match.groups()
    return _provenance_text_contains_date(
        candidate, f"{year}-{month}-{day}"
    ) and _provenance_text_contains_time(candidate, f"{hour}:{minute}")


def _provenance_text_contains_number(candidate: str, expected: Any) -> bool:
    """Match one finite typed number in Arabic or Japanese 0--59 notation."""

    if isinstance(expected, bool) or not isinstance(expected, (int, float)):
        return False
    if isinstance(expected, float) and not isfinite(expected):
        return False
    for match in ARABIC_NUMBER_PATTERN.finditer(candidate):
        token = match.group(0).replace(",", "")
        if float(expected) == 0.0 and token != "0":
            continue
        if float(token) == float(expected):
            return True
    if not float(expected).is_integer():
        return False
    forms = _japanese_number_forms(int(expected))
    alternatives = "|".join(
        re.escape(form) for form in sorted(forms, key=len, reverse=True)
    )
    pattern = rf"(?<![{TIME_NUMBER_BOUNDARY}])(?:{alternatives})(?![{TIME_NUMBER_BOUNDARY}])"
    return bool(forms) and re.search(pattern, candidate) is not None


def is_operational_enum_value(column: str, value: Any) -> bool:
    """Return whether ``value`` is evaluator-declared vocabulary for ``column``."""

    return not isinstance(value, bool) and value in OPERATIONAL_ENUM_VALUES.get(column, ())


def provenance_text_contains_typed_value(
    candidate: Any,
    expected: Any,
    contracts: list[tuple[str, str]],
) -> bool:
    """Return whether provenance text contains a typed value under ``contracts``.

    ``candidate`` must be text and ``expected`` a string or finite typed number.
    Embedded date/time/number occurrences and Japanese spoken phone digits are
    accepted; invalid types return ``False``. Exact-match helpers remain unchanged.
    """

    if not isinstance(candidate, str):
        return False
    normalized_candidate = unicodedata.normalize("NFKC", candidate)
    for column, value_type in contracts:
        if value_type in {"integer", "number"} and _provenance_text_contains_number(
            normalized_candidate, expected
        ):
            return True
        if not isinstance(expected, str):
            continue
        if _provenance_text_contains_datetime(normalized_candidate, expected):
            return True
        if value_type == "date" or column == "birthdate" or column.endswith("_date"):
            if _provenance_text_contains_date(normalized_candidate, expected):
                return True
        elif value_type == "time" or column == "time" or column.endswith("_time"):
            if _provenance_text_contains_time(normalized_candidate, expected):
                return True
        elif "phone" in column:
            normalized_expected = _provenance_phone(expected)
            if normalized_expected and normalized_expected in _provenance_phone(candidate):
                return True
        else:
            normalized_expected = _normalized_match_value(expected, column, value_type)
            normalized_text = _normalized_match_value(candidate, column, value_type)
            if (
                isinstance(normalized_expected, str)
                and normalized_expected
                and isinstance(normalized_text, str)
                and normalized_expected in normalized_text
            ):
                return True
            if _compact_provenance_text(expected) in _compact_provenance_text(candidate):
                return True
            if column == "plan_name" and _plan_name_prefix_matches(
                _compact_provenance_text(expected), _compact_provenance_text(candidate)
            ):
                return True
    return isinstance(expected, str) and expected in candidate


def _plan_name_prefix_matches(expected: str, candidate: str) -> bool:
    """Accept a spoken plan prefix unless the text goes on to name a different plan.

    After the shared prefix, the candidate may end or continue with hiragana (a
    particle such as "に"); any other character must match the plan name, so
    "スタンダード20" does not supply "スタンダード30".
    """

    prefix = expected[:MIN_PLAN_NAME_PREFIX_LENGTH]
    if len(expected) < MIN_PLAN_NAME_PREFIX_LENGTH:
        return False
    start = candidate.find(prefix)
    while start >= 0:
        i, j = start + len(prefix), len(prefix)
        while i < len(candidate) and j < len(expected) and not "ぁ" <= candidate[i] <= "ゖ":
            if candidate[i] != expected[j]:
                break
            i, j = i + 1, j + 1
        else:
            return True
        start = candidate.find(prefix, start + 1)
    return False


def _provenance_phone(value: str) -> str:
    """Normalize NFKC phone text with the supported Japanese digit readings."""

    normalized = unicodedata.normalize("NFKC", value)
    for spoken, digit in SPOKEN_PHONE_DIGITS:
        normalized = normalized.replace(spoken, digit)
    return PROVENANCE_PHONE_SEPARATOR_PATTERN.sub("", normalized)
