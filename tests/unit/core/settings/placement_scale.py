from __future__ import annotations

import pytest


def test_mask_norm_scale_matches_metric_range_scale(config) -> None:
    """Plan Phase 0.1: the mask budget norm (metrics.target_density) must equal
    the metric normalisation max (metrics.target_density_ranges.*.max), otherwise
    the mask caps density below the reward target and the policy is starved."""
    density = config.metrics.target_density
    ranges = config.metrics.target_density_ranges
    for key in ("enemy", "coin", "pipe", "platform", "block"):
        assert getattr(density, key) == pytest.approx(getattr(ranges, key).max), key
