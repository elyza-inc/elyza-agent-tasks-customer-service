"""Canonical and legacy evaluation variant contracts."""

import pytest

from elyza_agent_tasks_customer_service.evaluation.contracts.variants import (
    normalize_variant,
)


def test_unknown_variant_is_rejected() -> None:
    with pytest.raises(ValueError, match="baseline.*hard"):
        normalize_variant("unknown")
