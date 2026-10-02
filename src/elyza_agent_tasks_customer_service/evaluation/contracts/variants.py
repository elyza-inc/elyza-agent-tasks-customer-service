"""Canonical evaluation variant identifiers."""

from __future__ import annotations


ACCEPTED_VARIANT_IDS = ("baseline", "hard")


def normalize_variant(variant_id: object) -> str:
    """Return ``variant_id`` when it is ``baseline`` or ``hard``; otherwise raise ``ValueError``."""

    if variant_id not in ACCEPTED_VARIANT_IDS:
        raise ValueError(f"variant_id must be one of {list(ACCEPTED_VARIANT_IDS)}")
    return variant_id
