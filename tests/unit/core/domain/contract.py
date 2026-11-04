from __future__ import annotations

import numpy as np
import pytest
from pcgrl.core.domain.contract import (
    EpisodeState,
    RewardBreakdown,
    StepState,
    ValidationResult,
    Violation,
)
from pcgrl.core.domain.level import AreaType, LevelGrid
from pydantic import ValidationError


def _make_level() -> LevelGrid:
    """ make level."""
    tiles = np.zeros((4, 8), dtype=np.int16)
    return LevelGrid(
        tiles=tiles, width_tiles=8, height_tiles=4, tile_size_px=16, area_type=AreaType.OVERWORLD
    )


def test_violation_frozen() -> None:
    """Verify violation frozen."""
    v = Violation(rule="x", severity="hard", message="oops")
    with pytest.raises(ValidationError):
        v.rule = "y"


def test_violation_default_cells_empty() -> None:
    """Verify violation default cells empty."""
    v = Violation(rule="x", severity="soft", message="m")
    assert v.cells == ()


def test_validation_result_valid() -> None:
    """Verify validation result valid."""
    r = ValidationResult(is_valid=True)
    assert r.penalty == 0.0
    assert r.violations == ()


def test_validation_result_invalid_with_penalty() -> None:
    """Verify validation result invalid with penalty."""
    v = Violation(rule="r", severity="hard", message="bad")
    r = ValidationResult(is_valid=False, violations=(v,), penalty=1.0)
    assert not r.is_valid
    assert len(r.violations) == 1


def test_reward_breakdown_total() -> None:
    """Verify step state frozen."""
    rb = RewardBreakdown(total=2.5, components={"a": 1.5, "b": 1.0})
    assert rb.total == pytest.approx(2.5)
    assert rb.components["a"] == pytest.approx(1.5)


def test_step_state_frozen() -> None:
    """Verify step state frozen."""
    level = _make_level()
    state = StepState(
        level=level, cursor=(0, 0), step_count=0, target=np.zeros(8, dtype=np.float32)
    )
    with pytest.raises(ValidationError):
        state.step_count = 1


def test_episode_state_optional_gameplay() -> None:
    """Verify episode state optional gameplay."""
    level = _make_level()
    vr = ValidationResult(is_valid=True)
    ep = EpisodeState(level=level, target=np.zeros(8, dtype=np.float32), validator_result=vr)
    assert ep.gameplay_result is None
    assert ep.step_count == 0
