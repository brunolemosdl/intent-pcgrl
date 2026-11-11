from __future__ import annotations

import csv
import json

import numpy as np
import pytest
from pcgrl.app.experiments.spec import FactorialSpec, FactorSpec, apply_overrides
from pcgrl.core.helpers.path import resolve_path


def _spec(replicates: int = 2) -> FactorialSpec:
    """Two-factor toy spec for design and analyze tests."""
    return FactorialSpec(
        name="toy",
        kind="training",
        replicates=replicates,
        episodes=1,
        factors=[
            FactorSpec(
                name="A",
                overrides_low={"reward.composer.weight_style": 0.0},
                overrides_high={"reward.composer.weight_style": 2.0},
            ),
            FactorSpec(
                name="B",
                overrides_low={"environment.composer.intent_budget_mask": False},
                overrides_high={"environment.composer.intent_budget_mask": True},
            ),
        ],
        responses=["intent_error_mean"],
    )


def test_design_matrix_is_full_factorial_with_replicates() -> None:
    """2 factors x r=3 -> 12 rows in +-1 coding with merged overrides."""
    from pcgrl.app.experiments.factorial import design_matrix

    rows = design_matrix(_spec(replicates=3))
    assert len(rows) == 4 * 3
    cells = {row["cell"] for row in rows}
    assert cells == {"A-B-", "A-B+", "A+B-", "A+B+"}
    high = next(row for row in rows if row["cell"] == "A+B+")
    assert high["overrides"]["reward.composer.weight_style"] == 2.0
    assert high["overrides"]["environment.composer.intent_budget_mask"] is True
    assert high["levels"] == {"A": 1, "B": 1}


def test_spec_rejects_single_replicate() -> None:
    """r=1 has no error term and must be rejected."""
    with pytest.raises(ValueError, match="replicates"):
        _spec(replicates=1)


def test_apply_overrides_roundtrip(config) -> None:
    """Dotted-path overrides rebuild nested frozen models without mutation."""
    updated = apply_overrides(
        config,
        {
            "reward.composer.weight_style": 0.0,
            "environment.composer.duplicate_window": 0,
            "training.total_steps": 1234,
        },
    )
    assert updated.reward.composer.weight_style == 0.0
    assert updated.environment.composer.duplicate_window == 0
    assert updated.training.total_steps == 1234
    assert config.reward.composer.weight_style == 2.0
    assert config.environment.composer.duplicate_window == 3
    with pytest.raises(AttributeError, match="no field"):
        apply_overrides(config, {"reward.composer.not_a_field": 1.0})


def test_factorial_spec_yaml_loads() -> None:
    """The committed training factorial spec parses and validates."""
    spec = FactorialSpec.load(resolve_path("configs/experiments/factorial_composer.yaml"))
    assert spec.name == "factorial_composer"
    assert len(spec.factors) == 4
    assert spec.replicates >= 2


def test_analyze_stage_recovers_planted_effect(config, tmp_path) -> None:
    """A synthetic cells.csv with a planted A effect ranks A first."""
    from pcgrl.app.experiments.factorial import design_matrix, run_factorial

    spec = _spec()
    rng = np.random.default_rng(0)
    rows = []
    for row in design_matrix(spec):
        response = 0.5 + 0.3 * row["levels"]["A"] + rng.normal(0.0, 0.01)
        rows.append(
            {
                "cell": row["cell"],
                "replicate": row["replicate"],
                "level_A": row["levels"]["A"],
                "level_B": row["levels"]["B"],
                "intent_error_mean": round(float(response), 4),
                "n": 1,
            }
        )
    with (tmp_path / "cells.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    run_factorial(config, spec, stage="analyze", output_dir=tmp_path)
    anova = json.loads((tmp_path / "anova.json").read_text(encoding="utf-8"))
    effects = anova["intent_error_mean"]
    assert effects[0]["name"] == "A"
    assert effects[0]["effect"] == pytest.approx(0.6, abs=0.05)
    assert (tmp_path / "effects.csv").is_file()
    assert (tmp_path / "spec_snapshot.yaml").is_file()


def test_intent_budget_mask_toggle_disables_budget(config) -> None:
    """With the toggle off, the legal mask ignores intent budgets."""
    from unittest.mock import MagicMock

    from pcgrl.runtime.environment.composer_state import ComposerCore, ComposeState

    library_root = resolve_path(config.library.output_root)
    if not (library_root / "chunk_library.npz").is_file():
        pytest.skip("Chunk library not built")
    from pcgrl.core.library.types import ChunkLibrary
    from pcgrl.core.tiles.catalog import load_catalog

    library = ChunkLibrary.load(library_root)
    catalog = load_catalog(config)
    core_on = ComposerCore(library, config, catalog)
    off_config = apply_overrides(config, {"environment.composer.intent_budget_mask": False})
    core_off = ComposerCore(library, off_config, catalog)
    core_off._budget_mask = MagicMock()  # must never be consulted
    target = np.zeros(config.target_vector_dim, dtype=np.float32)
    state = ComposeState()
    mask_on = core_on.legal_chunk_mask(state, target)
    mask_off = core_off.legal_chunk_mask(state, target)
    core_off._budget_mask.assert_not_called()
    # A zero target makes budgets maximally restrictive: the toggled-off mask
    # must be at least as permissive as the budget-constrained one.
    assert mask_off.sum() >= mask_on.sum()


def test_split_cell_params_separates_runner_keys() -> None:
    """param.* keys leave the config-override dict."""
    from pcgrl.app.experiments.spec import split_cell_params

    overrides = {
        "translator.grounding": False,
        "param.description_style": "templated",
    }
    config_overrides, params = split_cell_params(overrides)
    assert config_overrides == {"translator.grounding": False}
    assert params == {"description_style": "templated"}


def test_inference_spec_yaml_loads() -> None:
    """The committed inference factorial spec parses and validates."""
    from pcgrl.app.experiments.spec import FactorialSpec

    spec = FactorialSpec.load(resolve_path("configs/experiments/factorial_inference.yaml"))
    assert spec.kind == "inference"
    assert spec.base_overrides["generation.require_gameplay_bridge"] is False
    assert len(spec.factors) == 3
