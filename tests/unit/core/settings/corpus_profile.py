from __future__ import annotations

from pcgrl.core.corpus.profile import CorpusProfile
from pcgrl.core.settings.loader import apply_corpus_profile, load_config


def _fake_profile() -> CorpusProfile:
    """ fake profile."""
    return CorpusProfile(
        {
            "n_levels": 5,
            "density": {
                "coin": {"p95": 0.005, "max": 0.006},
                "platform": {"p95": 0.12, "max": 0.15},
                "enemy": {"p95": 0.01, "max": 0.011},
                "pipe": {"p95": 0.03, "max": 0.04},
                "block": {"p95": 0.02, "max": 0.03},
            },
            "height_above_surface": {
                "floating_platform": {"hist": [0, 0, 0.05, 0.2, 1.0, 0.3], "pcts": {"p90": 8}},
                "coin": {"pcts": {"p90": 7}, "arc_over_pit_frac": 0.09},
                "block": {"pcts": {"p90": 9}},
            },
            "gaps": {
                "width": {"pcts": {"p90": 5}},
                "count_per_level_norm": {"pcts": {"p90": 8}},
            },
            "pipe": {"spacing": {"pcts": {"p25": 4}}},
        }
    )


def test_apply_corpus_profile_derives_values() -> None:
    """Verify apply corpus profile derives values."""
    base = load_config(training="initial", apply_profile=False)
    out = apply_corpus_profile(base, _fake_profile())
    # densities come from the chosen percentile (default p95) and stay in sync.
    assert out.metrics.target_density.coin == 0.005
    assert out.metrics.target_density_ranges.coin.max == 0.005
    assert out.metrics.target_density.platform == 0.12
    # height bands from p90.
    assert out.metrics.semantic.coin_max_height_above_surface == 7
    assert out.metrics.semantic.block_max_height_above_surface == 9
    # gap + pipe rules.
    assert out.validation.max_gap_width == 5
    assert out.validation.max_gap_count == 8
    assert out.validation.pipe_min_gap == 4
    # min floating height = smallest hist index >= threshold (0.10 -> idx 3).
    assert out.validation.min_floating_platform_height == 3


def test_apply_corpus_profile_noop_when_unavailable() -> None:
    """Verify apply corpus profile noop when unavailable."""
    base = load_config(training="initial", apply_profile=False)
    out = apply_corpus_profile(base, CorpusProfile(None))
    assert out is base  # unchanged when the profile is missing


def test_apply_corpus_profile_respects_disabled_flag() -> None:
    """Verify apply corpus profile respects disabled flag."""
    base = load_config(training="initial", apply_profile=False)
    disabled = base.model_copy(
        update={
            "metrics": base.metrics.model_copy(
                update={
                    "corpus_profile": base.metrics.corpus_profile.model_copy(
                        update={"enabled": False}
                    )
                }
            )
        }
    )
    out = apply_corpus_profile(disabled, _fake_profile())
    assert out.metrics.target_density.coin == disabled.metrics.target_density.coin


def test_corpus_profile_missing_file_unavailable(tmp_path) -> None:
    """Verify corpus profile missing file unavailable."""
    prof = CorpusProfile.load(tmp_path)
    assert not prof.available
    assert prof.density("coin", 95) is None
    assert prof.height_curve("coin") is None
