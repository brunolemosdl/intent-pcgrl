from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np

PATTERNS_FILENAME = "pattern_distributions.npz"
_EPSILON = 1e-05


def count_patterns(tiles: np.ndarray, size: int) -> Counter[bytes]:
    """Count every ``size x size`` window in a tile grid.

    Args:
        tiles: Level tile id grid.
        size: Pattern window side length.

    Returns:
        Counter mapping raw window bytes to occurrence counts.
    """
    height, width = tiles.shape
    counts: Counter[bytes] = Counter()
    if height < size or width < size:
        return counts
    grid = np.ascontiguousarray(tiles, dtype=np.int16)
    windows = np.lib.stride_tricks.sliding_window_view(grid, (size, size))
    # Flatten every window into one contiguous row and dedupe in C instead of
    # one .tobytes() per window: same keys, ~10x fewer Python-level ops.
    flat = np.ascontiguousarray(windows.reshape(-1, size * size))
    unique, occurrences = np.unique(flat, axis=0, return_counts=True)
    for row, count in zip(unique, occurrences, strict=True):
        counts[row.tobytes()] = int(count)
    return counts


def empty_pattern_key(empty_id: int, size: int) -> bytes:
    """Return the byte key of the all-empty ``size x size`` pattern.

    Args:
        empty_id: Catalog id for empty tiles.
        size: Pattern window side length.

    Returns:
        Raw bytes key used by ``exclude`` in TPKL scoring.
    """
    return np.full((size, size), empty_id, dtype=np.int16).tobytes()


class PatternReference:
    """Corpus tile-pattern distributions for one split, indexed by size.

    Description:
        Produced by the ``prepare`` stage as ``pattern_distributions.npz``. The
        reward must load split="train"; evaluation loads split="heldout" — the
        anti-Goodhart separation between optimisation and judgment.

    """

    def __init__(self, distributions: dict[int, dict[bytes, float]]) -> None:
        """Normalise raw counts into probabilities per pattern size.

        Args:
            distributions: Map from pattern size to count dict.

        Raises:
            ValueError: When any size has zero total count.
        """
        self._distributions: dict[int, dict[bytes, float]] = {}
        for size, counts in distributions.items():
            total = float(sum(counts.values()))
            if total <= 0.0:
                raise ValueError(f"pattern distribution for k={size} is empty")
            self._distributions[int(size)] = {
                key: value / total for key, value in counts.items()
            }

    @property
    def sizes(self) -> tuple[int, ...]:
        """Pattern sizes available in this reference."""
        return tuple(sorted(self._distributions))

    def probabilities(
        self, size: int, *, exclude: bytes | None = None
    ) -> dict[bytes, float]:
        """Return pattern probabilities for one window size.

        Args:
            size: Pattern window side length.
            exclude: Optional pattern key to drop and renormalize.

        Returns:
            Probability map over pattern byte keys.

        Raises:
            KeyError: When ``size`` is not in the reference.
        """
        if size not in self._distributions:
            raise KeyError(
                f"no pattern distribution for k={size} (available: {self.sizes})"
            )
        distribution = self._distributions[size]
        if exclude is None or exclude not in distribution:
            return distribution
        remaining = {key: value for key, value in distribution.items() if key != exclude}
        total = sum(remaining.values())
        if total <= 0.0:
            return {}
        return {key: value / total for key, value in remaining.items()}

    @classmethod
    def load(
        cls, processed_root: str | Path, *, split: str = "train"
    ) -> PatternReference:
        """Load one split of ``pattern_distributions.npz``.

        Args:
            processed_root: Processed dataset root directory.
            split: ``train`` or ``heldout``.

        Returns:
            ``PatternReference`` for the requested split.

        Raises:
            ValueError: When ``split`` is not ``train`` or ``heldout``.
            FileNotFoundError: When the npz artifact is missing.
        """
        if split not in ("train", "heldout"):
            raise ValueError(f"split must be 'train' or 'heldout', got {split!r}")
        path = Path(processed_root) / PATTERNS_FILENAME
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found; run the 'prepare' stage to compute pattern distributions"
            )
        distributions: dict[int, dict[bytes, float]] = {}
        with np.load(path) as data:
            for size in data["sizes"].tolist():
                patterns = data[f"{split}_patterns_k{size}"]
                counts = data[f"{split}_counts_k{size}"]
                distributions[int(size)] = {
                    np.ascontiguousarray(patterns[index], dtype=np.int16).tobytes(): float(
                        counts[index]
                    )
                    for index in range(patterns.shape[0])
                }
        return cls(distributions)


def tile_pattern_kl(
    tiles: np.ndarray,
    reference: PatternReference,
    size: int,
    *,
    epsilon: float = _EPSILON,
    exclude: bytes | None = None,
) -> float:
    """Compute KL(P_level || P_reference) over ``size x size`` patterns.

    Args:
        tiles: Level tile id grid.
        reference: Corpus pattern reference.
        size: Pattern window side length.
        epsilon: Smoothing floor for unseen reference patterns.
        exclude: Pattern key excluded from both sides (typically all-empty).

    Returns:
        Non-negative KL divergence, or ``inf`` when no patterns remain.

    Raises:
        ValueError: When the level is smaller than the pattern window and
            ``exclude`` is not set.
    """
    level_counts = count_patterns(tiles, size)
    if exclude is not None:
        level_counts.pop(exclude, None)
    total = float(sum(level_counts.values()))
    if total <= 0.0:
        if exclude is not None:
            return math.inf
        raise ValueError(f"level is smaller than the {size}x{size} pattern window")
    reference_probabilities = reference.probabilities(size, exclude=exclude)
    divergence = 0.0
    for key, value in level_counts.items():
        p = value / total
        q = reference_probabilities.get(key, 0.0) + epsilon
        divergence += p * math.log(p / q)
    return max(0.0, divergence)


def style_score(
    tiles: np.ndarray,
    reference: PatternReference,
    size: int,
    *,
    epsilon: float = _EPSILON,
    exclude: bytes | None = None,
) -> float:
    """Return ``exp(-TPKL)`` style score in ``[0, 1]``.

    Args:
        tiles: Level tile id grid.
        reference: Corpus pattern reference.
        size: Pattern window side length.
        epsilon: KL smoothing floor.
        exclude: Pattern key to exclude from both distributions.

    Returns:
        Style score where ``1.0`` matches the reference distribution.
    """
    divergence = tile_pattern_kl(tiles, reference, size, epsilon=epsilon, exclude=exclude)
    if math.isinf(divergence):
        return 0.0
    return math.exp(-divergence)
