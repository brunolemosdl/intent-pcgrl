from __future__ import annotations

import csv
import json

import pytest
from pcgrl.core.helpers.path import resolve_path


def _require_artifacts(config) -> None:
    """Skip when the prepare/library artifacts are not on disk."""
    processed = resolve_path(config.dataset.processed_root)
    library = resolve_path(config.library.output_root)
    if (
        not (processed / "pattern_distributions.npz").is_file()
        or not (library / "chunk_library.npz").is_file()
    ):
        pytest.skip("Processed corpus / chunk library not built")


def test_build_generate_fn_rejects_unknown_system(config) -> None:
    """Unknown system names raise before any artifact is touched."""
    from pcgrl.app.evaluation.comparison import build_generate_fn

    with pytest.raises(ValueError, match="unknown system"):
        build_generate_fn(config, "definitely_not_a_system")


@pytest.mark.parametrize(
    "system",
    ["ppo", "ppo_best_of_n", "ppo_best_of_n_polished", "ppo_random_edits", "ppo_planner_edits"],
)
def test_build_generate_fn_requires_checkpoint_for_ppo(config, system) -> None:
    """The ppo systems demand a checkpoint."""
    from pcgrl.app.evaluation.comparison import build_generate_fn

    with pytest.raises(ValueError, match="checkpoint"):
        build_generate_fn(config, system)


def test_run_comparison_writes_all_fragments(config, tmp_path) -> None:
    """One real run over light systems yields the four report fragments."""
    _require_artifacts(config)
    from pcgrl.app.evaluation.comparison import run_comparison

    csv_path = run_comparison(
        config,
        systems=["random_legal", "corpus_replay"],
        episodes=1,
        output_dir=tmp_path,
    )
    assert csv_path.is_file()
    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    cases = len(config.evaluation.target_suite)
    assert len(rows) == 2 * cases
    expected_columns = {
        "system",
        "case",
        "seed",
        "episode",
        "target_source",
        "wall_time_s",
        "terminal_reward",
        "component_alignment",
        "component_style",
        "component_playability",
        "component_duplicate",
        "intent_error_mean",
        "intent_error_max",
        "duplicate_rate",
        "error_enemy_density",
        "error_platform_usage",
        "tpkl_heldout_k3",
        "census_gap_widths",
    }
    assert expected_columns.issubset(rows[0].keys())
    replay_rows = [row for row in rows if row["system"] == "corpus_replay"]
    # corpus_replay has no chunk ids: duplicate terms are 0 by construction.
    assert all(float(row["component_duplicate"]) == 0.0 for row in replay_rows)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert set(summary) == {"random_legal", "corpus_replay"}
    assert "terminal_reward" in summary["random_legal"]
    pairwise = json.loads((tmp_path / "pairwise.json").read_text(encoding="utf-8"))
    assert any(item["metric"] == "terminal_reward" for item in pairwise)
    assert (tmp_path / "comparison.md").is_file()
    # Same seeds for every system: the protocol is paired by construction.
    seeds_by_system = {
        system: sorted(int(row["seed"]) for row in rows if row["system"] == system)
        for system in ("random_legal", "corpus_replay")
    }
    assert seeds_by_system["random_legal"] == seeds_by_system["corpus_replay"]


def test_run_comparison_with_fake_target_source(config, tmp_path) -> None:
    """An NL-fed run carries per-stage error columns scored against the oracle."""
    _require_artifacts(config)
    from pcgrl.app.evaluation.comparison import run_comparison

    csv_path = run_comparison(
        config,
        systems=["corpus_replay"],
        episodes=1,
        target_source="fake",
        output_dir=tmp_path,
    )
    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    for row in rows:
        assert row["target_source"] == "fake"
        assert "translation_error_mean" in row
        assert "policy_error_mean" in row
        assert float(row["translation_error_mean"]) >= 0.0
