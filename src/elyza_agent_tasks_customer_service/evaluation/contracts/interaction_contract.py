"""Evaluation-side pairing contract for M16/M19 measurements."""

from __future__ import annotations

from typing import Any


def interaction_event_refs(question_contract: dict[str, Any]) -> set[str]:
    """Return all event refs used by fixed interaction opportunities.

    ``question_contract`` must be a validated
    ``interaction_closed_questions`` object.
    """

    refs: set[str] = set()
    for instance in question_contract["instances"]:
        for key in ("candidate_span_refs", "source_refs"):
            refs.update(instance[key])
        refs.update(
            value
            for value in instance["parameters"].values()
            if isinstance(value, str)
        )
    return refs
