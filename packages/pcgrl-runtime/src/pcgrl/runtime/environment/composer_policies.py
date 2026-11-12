from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from pcgrl.core.library.types import ChunkLibrary
from pcgrl.runtime.environment.composer_state import ComposerCore, ComposeState
from pcgrl.runtime.reward.composer_terminal import ComposerTerminalReward

BASELINE_NAMES = ("random_legal", "greedy_intent", "beam_search", "markov")

_PROXY_DIMS = 4


def _proxy_alignment(core: ComposerCore, state: ComposeState, target: np.ndarray) -> float:
    """Cheap alignment proxy over the four count-steerable density dims.

    Uses the same normalisation bands as the real intent metrics but no
    stitching or flood, so greedy/beam stay O(K) per decision.
    """
    config = core.config
    area = config.game.width_tiles * config.game.height_tiles
    ranges = config.metrics.target_density_ranges
    maxima = np.array(
        [ranges.enemy.max, ranges.coin.max, ranges.block.max, ranges.pipe.max],
        dtype=np.float64,
    )
    densities = np.asarray(state.counts, dtype=np.float64) / area
    measured = np.clip(densities / np.maximum(maxima, 1e-09), 0.0, 1.0)
    deltas = np.abs(measured - np.asarray(target[:_PROXY_DIMS], dtype=np.float64))
    return 1.0 - float(deltas.mean())


def rollout_random_legal(
    core: ComposerCore, target: np.ndarray, rng: np.random.Generator
) -> ComposeState:
    """Compose by sampling uniformly over the legal mask at every step.

    Args:
        core: Composer core with library and masks.
        target: Intent target vector.
        rng: Random generator.

    Returns:
        Terminal composition state after the rollout.
    """
    state = ComposeState()
    while not core.is_terminal(state):
        mask = core.legal_chunk_mask(state, target)
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            break
        state = core.apply_chunk(state, int(rng.choice(legal)))
    return state


def rollout_greedy_intent(
    core: ComposerCore, target: np.ndarray, rng: np.random.Generator
) -> ComposeState:
    """Compose by one-step lookahead on the alignment proxy.

    Args:
        core: Composer core with library and masks.
        target: Intent target vector.
        rng: Random generator for tie-breaking.

    Returns:
        Terminal composition state after greedy rollout.
    """
    state = ComposeState()
    while not core.is_terminal(state):
        mask = core.legal_chunk_mask(state, target)
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            break
        scores = np.array(
            [
                _proxy_alignment(core, core.apply_chunk(state, int(chunk_id)), target)
                for chunk_id in legal
            ]
        )
        best = legal[np.flatnonzero(scores == scores.max())]
        state = core.apply_chunk(state, int(rng.choice(best)))
    return state


def rollout_beam_search(
    core: ComposerCore,
    target: np.ndarray,
    rng: np.random.Generator,
    *,
    beam_width: int = 8,
    terminal: ComposerTerminalReward | None = None,
) -> ComposeState:
    """Width-B beam over the proxy score with real-reward final re-ranking.

    Description:
        The re-ranking pool is capped at twice the beam width: the real terminal
        reward needs a stitch plus two floods per candidate, so only the
        proxy-best completions receive it. Without a terminal computer the proxy
        decides.

    Args:
        core: See signature.
        target: See signature.
        rng: See signature.
        beam_width: See signature.
        terminal: See signature.

    Returns:
        ComposeState: Result of the operation.
    """
    beams: list[ComposeState] = [ComposeState()]
    completed: list[tuple[float, ComposeState]] = []
    max_decisions = core.config.game.width_tiles // max(1, min(core.library.widths())) + 2
    for _ in range(max_decisions):
        candidates: list[tuple[float, float, ComposeState]] = []
        for state in beams:
            mask = core.legal_chunk_mask(state, target)
            for chunk_id in np.flatnonzero(mask):
                child = core.apply_chunk(state, int(chunk_id))
                score = _proxy_alignment(core, child, target)
                candidates.append((score, rng.random(), child))
        if not candidates:
            break
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        beams = []
        for score, _, child in candidates:
            if core.is_terminal(child):
                completed.append((score, child))
            elif len(beams) < beam_width:
                beams.append(child)
        if not beams:
            break
    completed.sort(key=lambda item: item[0], reverse=True)
    pool = [child for _, child in completed[: 2 * beam_width]] or beams
    if not pool:
        return ComposeState()
    if terminal is not None:
        rewards = [
            terminal.compute(core.stitch_level(state), target, chunk_ids=state.chunk_ids)[0]
            for state in pool
        ]
        return pool[int(np.argmax(rewards))]
    return max(pool, key=lambda state: _proxy_alignment(core, state, target))


@dataclass(frozen=True)
class BigramMarkov:
    """Chunk bigram model fitted on per-level chunk sequences.

    Description:
        Style-only baseline: knows corpus chunk transitions but not intent.

    Attributes:
        transitions: Outgoing chunk weights keyed by previous chunk id.
        initial: Weights for the first chunk in a level.
    """

    transitions: dict[int, dict[int, float]]
    initial: dict[int, float]

    @classmethod
    def fit(cls, library: ChunkLibrary) -> BigramMarkov:
        """Fit bigram counts from chunk provenance grouped by source level.

        Args:
            library: Annotated chunk library.

        Returns:
            Fitted ``BigramMarkov`` model.
        """
        sequences: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for chunk in library.chunks:
            sequences[chunk.level_id].append((chunk.column_offset, chunk.chunk_id))
        transitions: dict[int, dict[int, float]] = defaultdict(dict)
        initial: dict[int, float] = {}
        for entries in sequences.values():
            ordered = [chunk_id for _, chunk_id in sorted(entries)]
            if not ordered:
                continue
            initial[ordered[0]] = initial.get(ordered[0], 0.0) + 1.0
            for previous, current in itertools.pairwise(ordered):
                row = transitions[previous]
                row[current] = row.get(current, 0.0) + 1.0
        return cls(transitions=dict(transitions), initial=initial)

    def _sample(
        self,
        weights: dict[int, float],
        legal: np.ndarray,
        rng: np.random.Generator,
    ) -> int:
        """Sample a legal chunk by bigram weight, uniform when unseen."""
        probabilities = np.array([weights.get(int(c), 0.0) for c in legal])
        total = probabilities.sum()
        if total <= 0.0:
            return int(rng.choice(legal))
        return int(rng.choice(legal, p=probabilities / total))

    def rollout(
        self, core: ComposerCore, target: np.ndarray, rng: np.random.Generator
    ) -> ComposeState:
        """Compose by sampling bigram transitions subject to the legal mask."""
        state = ComposeState()
        while not core.is_terminal(state):
            mask = core.legal_chunk_mask(state, target)
            legal = np.flatnonzero(mask)
            if legal.size == 0:
                break
            if state.num_decisions == 0:
                weights = self.initial
            else:
                weights = self.transitions.get(state.chunk_ids[-1], {})
            state = core.apply_chunk(state, self._sample(weights, legal, rng))
        return state


def rollout_baseline(
    name: str,
    core: ComposerCore,
    target: np.ndarray,
    rng: np.random.Generator,
    *,
    terminal: ComposerTerminalReward | None = None,
    markov: BigramMarkov | None = None,
    beam_width: int = 8,
) -> ComposeState:
    """Dispatch a baseline rollout by name.

    Args:
        name: One of ``BASELINE_NAMES``.
        core: Composer core shared by all baselines.
        target: Intent target vector.
        rng: Random generator.
        terminal: Optional terminal reward for beam re-ranking.
        markov: Optional pre-fitted bigram model.
        beam_width: Beam width for ``beam_search``.

    Returns:
        Terminal composition state.

    Raises:
        ValueError: When ``name`` is not a known baseline.
    """
    if name == "random_legal":
        return rollout_random_legal(core, target, rng)
    if name == "greedy_intent":
        return rollout_greedy_intent(core, target, rng)
    if name == "beam_search":
        return rollout_beam_search(core, target, rng, beam_width=beam_width, terminal=terminal)
    if name == "markov":
        model = markov or BigramMarkov.fit(core.library)
        return model.rollout(core, target, rng)
    raise ValueError(f"unknown baseline {name!r} (choose from {BASELINE_NAMES})")
