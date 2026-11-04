from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pcgrl.runtime.environment.composer_state import ComposerCore, ComposeState
from pcgrl.runtime.metrics.intent import INTENT_KEYS, compute_intent_vector

VIABILITY_FILENAME = "intent_viability.json"

_UNIFORM_SHARE = 0.2
_JITTER = 0.08


class IntentViabilityMap:
    """Empirical cloud of intents realisable by random-legal composition.

    Description:
        Training targets are sampled from the
        REACHABLE manifold (stored samples plus jitter, preserving inter-dim
        correlations), with a uniform exploration share so unreachable corners
        still get probed and penalised honestly by the alignment term. Also the
        published artifact defining the announced controllable domain
        (per-dimension quantiles plus correlation matrix).

    """

    def __init__(self, samples: np.ndarray) -> None:
        """Validate and store the sample matrix.

        Args:
            samples: Array of shape ``(N, len(INTENT_KEYS))``.

        Raises:
            ValueError: When shape is wrong or fewer than two samples.
        """
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 2 or samples.shape[1] != len(INTENT_KEYS):
            raise ValueError(
                f"viability samples must be (N, {len(INTENT_KEYS)}), got {samples.shape}"
            )
        if samples.shape[0] < 2:
            raise ValueError("viability map needs at least 2 samples")
        self.samples = samples

    def quantiles(
        self, levels: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)
    ) -> dict[str, dict[str, float]]:
        """Compute per-dimension quantiles of the viable intent cloud.

        Args:
            levels: Quantile probabilities to evaluate.

        Returns:
            Nested dict keyed by intent dimension then ``pXX`` label.
        """
        result: dict[str, dict[str, float]] = {}
        for index, key in enumerate(INTENT_KEYS):
            column = self.samples[:, index]
            result[key] = {
                f"p{int(level * 100):02d}": float(np.quantile(column, level))
                for level in levels
            }
        return result

    def correlations(self) -> np.ndarray:
        """Pearson correlation matrix between intent dimensions.

        Returns:
            Square matrix of shape ``(dims, dims)`` with NaNs zeroed.
        """
        with np.errstate(invalid="ignore"):
            matrix = np.corrcoef(self.samples, rowvar=False)
        return np.nan_to_num(matrix, nan=0.0)  # type: ignore[no-any-return]

    def correlated_pairs(self, threshold: float = 0.8) -> list[tuple[str, str, float]]:
        """List dimension pairs with |correlation| above threshold.

        Args:
            threshold: Minimum absolute correlation to report.

        Returns:
            List of ``(dim_a, dim_b, correlation)`` tuples.
        """
        matrix = self.correlations()
        pairs = []
        for i in range(len(INTENT_KEYS)):
            for j in range(i + 1, len(INTENT_KEYS)):
                if abs(matrix[i, j]) > threshold:
                    pairs.append(
                        (INTENT_KEYS[i], INTENT_KEYS[j], float(matrix[i, j]))
                    )
        return pairs

    def sample_target(self, rng: np.random.Generator) -> np.ndarray:
        """Draw a training target from the viable manifold or uniform exploration.

        Args:
            rng: NumPy random generator.

        Returns:
            Intent vector in ``[0, 1]`` with shape ``(dims,)``.
        """
        if rng.random() < _UNIFORM_SHARE:
            return rng.uniform(0.0, 1.0, size=len(INTENT_KEYS)).astype(np.float32)
        base = self.samples[int(rng.integers(0, self.samples.shape[0]))]
        jitter = rng.uniform(-_JITTER, _JITTER, size=base.shape)
        return np.clip(base + jitter, 0.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]

    def save(self, directory: str | Path) -> Path:
        """Persist samples, quantiles, and correlated pairs as JSON.

        Args:
            directory: Output directory (created if missing).

        Returns:
            Path to the written ``intent_viability.json`` file.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / VIABILITY_FILENAME
        payload = {
            "keys": list(INTENT_KEYS),
            "samples": self.samples.tolist(),
            "quantiles": self.quantiles(),
            "correlated_pairs": [
                {"a": a, "b": b, "corr": corr} for a, b, corr in self.correlated_pairs()
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, directory: str | Path) -> IntentViabilityMap:
        """Load a saved viability map from disk.

        Args:
            directory: Directory containing ``intent_viability.json``.

        Returns:
            Reconstructed ``IntentViabilityMap``.

        Raises:
            FileNotFoundError: When the JSON artifact is missing.
            ValueError: When intent dimension keys do not match current schema.
        """
        path = Path(directory) / VIABILITY_FILENAME
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found; run the 'library' stage to build the viability map"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("keys") != list(INTENT_KEYS):
            raise ValueError(
                "viability map was built for different intent dimensions; "
                "re-run the 'library' stage"
            )
        return cls(np.asarray(payload["samples"], dtype=np.float32))


def build_viability_map(
    core: ComposerCore, *, episodes: int, seed: int
) -> IntentViabilityMap:
    """Build a viability map from random-legal composition rollouts.

    Args:
        core: Shared composer core (library + masks).
        episodes: Number of random episodes to sample.
        seed: RNG seed for reproducibility.

    Returns:
        ``IntentViabilityMap`` over successful rollouts.

    Raises:
        ValueError: When fewer than two successful rollouts complete.
    """
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(episodes):
        state = ComposeState()
        while not core.is_terminal(state):
            mask = core.legal_chunk_mask(state)
            legal = np.flatnonzero(mask)
            if legal.size == 0:
                break
            state = core.apply_chunk(state, int(rng.choice(legal)))
        if state.num_decisions == 0:
            continue
        level = core.stitch_level(state)
        samples.append(compute_intent_vector(level, core.catalog, core.config))
    if len(samples) < 2:
        raise ValueError("not enough successful rollouts to build a viability map")
    return IntentViabilityMap(np.stack(samples))
