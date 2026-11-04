from __future__ import annotations

import numpy as np
import pytest
from pcgrl.app.intent.fake import FakeIntentTranslator
from pcgrl.app.intent.verbalizer import (
    band_center,
    band_index,
    verbalize,
    verticality_thresholds,
)


def test_verbalize_is_deterministic_and_covers_all_dims(config) -> None:
    """The canonical sentence is byte-stable and mentions every dimension."""
    vector = np.array(config.default_target_vector(), dtype=np.float32)
    first = verbalize(vector, config)
    assert first == verbalize(vector, config)
    assert first.startswith("A level with ")
    assert first.count(",") >= 6


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_fake_inverts_verbalize_per_dimension_band(config, seed) -> None:
    """Property: fake(verbalize(v)) lands in v's band for every dimension."""
    rng = np.random.default_rng(seed)
    names = config.target_vector_field_names()
    raw = rng.uniform(0.0, 1.0, size=len(names))
    reference = np.array(
        [
            band_center(name, band_index(name, float(value), config), config)
            for name, value in zip(names, raw, strict=True)
        ],
        dtype=np.float32,
    )
    description = verbalize(reference, config)
    recovered = FakeIntentTranslator(config).translate(description).vector
    for index, name in enumerate(names):
        assert band_index(name, float(recovered[index]), config) == band_index(
            name, float(reference[index]), config
        ), f"{name}: {recovered[index]} vs {reference[index]} in {description!r}"


def test_verticality_thresholds_keep_fake_values_in_band(config) -> None:
    """The clamped thresholds keep the fake's 0.05/0.70/0.95 in their bands."""
    low, high = verticality_thresholds(config)
    assert 0.05 < low <= 0.70
    assert 0.70 < high <= 0.95


def test_translation_eval_smoke_on_fake(config, tmp_path) -> None:
    """The translation eval emits its CSV and summary fragments on the fake."""
    from pcgrl.app.experiments.llm_eval import run_translation_eval

    csv_path = run_translation_eval(
        config,
        FakeIntentTranslator(config),
        output_dir=tmp_path,
        templated_cases=6,
    )
    assert csv_path.is_file()
    import json

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_scored"] >= 6
    # Templated pairs are built from the fake's own phrase tables, so the
    # error on them must be small (band centers round-trip exactly).
    assert summary["translation_error_mean"] <= 0.2
    assert "enemy_density" in summary["per_dim"]


def test_monotonicity_smoke_on_stub_generator(config, tmp_path, blank_level) -> None:
    """The monotonicity experiment writes fragments with a stub generator."""
    from pcgrl.app.experiments.llm_eval import run_monotonicity

    csv_path = run_monotonicity(
        config,
        FakeIntentTranslator(config),
        lambda target, seed: blank_level,
        output_dir=tmp_path,
        episodes=1,
    )
    assert csv_path.is_file()
    assert (tmp_path / "monotonicity_summary.json").is_file()
