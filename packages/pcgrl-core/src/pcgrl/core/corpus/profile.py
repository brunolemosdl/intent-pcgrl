from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast


class CorpusProfile:
    """Read-only view over ``corpus_profile.json``.

    Description:
        Loads statistics produced by the corpus-profile extractor
        (``pcgrl.app.data.corpus.profile``). When the file is absent or
        unreadable, ``available`` is ``False`` and every accessor returns
        ``None`` so callers fall back to static YAML defaults.

    Attributes:
        available: Whether a non-empty profile was loaded successfully.
    """

    def __init__(self, data: dict[str, Any] | None) -> None:
        """Wrap parsed profile JSON; mark unavailable when empty or missing."""
        self._data = data or {}
        self.available = bool(data) and int(self._data.get("n_levels", 0)) > 0

    # -- construction ---------------------------------------------------------
    @classmethod
    def load(cls, processed_root: str | Path) -> CorpusProfile:
        """Load a profile from the processed dataset directory.

        Args:
            processed_root: Directory containing ``corpus_profile.json``.

        Returns:
            A profile instance; ``available`` is ``False`` when the file is
            missing or cannot be parsed.
        """
        path = Path(processed_root) / "corpus_profile.json"
        if not path.is_file():
            return cls(None)
        try:
            return cls(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            return cls(None)

    # -- scalar accessors -----------------------------------------------------
    def _pct(self, node: dict[str, Any] | None, pct: int) -> float | None:
        """Return the value at percentile key ``p{pct}`` or the node maximum."""
        if not node:
            return None
        key = f"p{pct}"
        if key in node:
            return float(node[key])
        return float(node["max"]) if "max" in node else None

    def density(self, name: str, pct: int) -> float | None:
        """Return per-area density of a category at the given percentile.

        Args:
            name: Density key (e.g. ``enemy``, ``coin``, ``pipe``).
            pct: Percentile in ``[0, 100]``.

        Returns:
            Density value, or ``None`` when the profile is unavailable.
        """
        if not self.available:
            return None
        return self._pct(self._data.get("density", {}).get(name), pct)

    def _height_node(self, kind: str) -> dict[str, Any] | None:
        """Return the height-above-surface node for a tile kind, if present."""
        nodes = self._data.get("height_above_surface", {})
        if not isinstance(nodes, dict):
            return None
        node = nodes.get(kind)
        return cast(dict[str, Any] | None, node if isinstance(node, dict) else None)

    def height_pct(self, kind: str, pct: int) -> float | None:
        """Return height-above-surface percentile for a tile kind.

        Args:
            kind: Height curve key (e.g. ``coin``, ``block``).
            pct: Percentile in ``[0, 100]``.

        Returns:
            Height value, or ``None`` when unavailable.
        """
        node = self._height_node(kind)
        return self._pct(node.get("pcts") if node else None, pct)

    def height_curve(self, kind: str) -> list[float] | None:
        """Return the normalised height-above-surface histogram.

        Args:
            kind: Height curve key (e.g. ``floating_platform``).

        Returns:
            Histogram where index is height and the mode equals ``1.0``, or
            ``None`` when unavailable.
        """
        if not self.available:
            return None
        node = self._height_node(kind)
        hist = node.get("hist") if node else None
        return list(hist) if hist else None

    def coin_arc_frac(self) -> float | None:
        """Return the fraction of coins placed in arc-over-pit patterns."""
        node = self._height_node("coin")
        return float(node["arc_over_pit_frac"]) if node and "arc_over_pit_frac" in node else None

    def gap_width(self, pct: int) -> float | None:
        """Return floor-gap width at the given percentile.

        Args:
            pct: Percentile in ``[0, 100]``.

        Returns:
            Gap width, or ``None`` when unavailable.
        """
        if not self.available:
            return None
        return self._pct(self._data.get("gaps", {}).get("width", {}).get("pcts"), pct)

    def gap_count(self, pct: int) -> float | None:
        """Return per-width-normalised gap count at the given percentile.

        Args:
            pct: Percentile in ``[0, 100]``.

        Returns:
            Gap count, or ``None`` when unavailable.
        """
        if not self.available:
            return None
        return self._pct(
            self._data.get("gaps", {}).get("count_per_level_norm", {}).get("pcts"), pct
        )

    def pipe_spacing(self, pct: int) -> float | None:
        """Return minimum horizontal pipe spacing at the given percentile.

        Args:
            pct: Percentile in ``[0, 100]``.

        Returns:
            Spacing in columns, or ``None`` when unavailable.
        """
        if not self.available:
            return None
        return self._pct(self._data.get("pipe", {}).get("spacing", {}).get("pcts"), pct)

    def min_curve_height(self, kind: str, threshold: float) -> int | None:
        """Return the smallest height whose histogram mass meets a threshold.

        Description:
            Used to derive the floating-platform floor dynamically; heights
            with negligible corpus mass are treated as degenerate.

        Args:
            kind: Height curve key (e.g. ``floating_platform``).
            threshold: Minimum normalised histogram value.

        Returns:
            Smallest qualifying height, or ``None`` when unavailable.
        """
        curve = self.height_curve(kind)
        if not curve:
            return None
        for h, v in enumerate(curve):
            if h >= 1 and v >= threshold:
                return h
        return None
