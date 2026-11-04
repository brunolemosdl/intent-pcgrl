from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from pcgrl.core.domain.level import LevelGrid


class Violation(BaseModel):
    """Single structural or semantic rule breach.

    Attributes:
        rule: Machine-readable rule identifier.
        severity: ``hard`` rejects the level; ``soft`` contributes penalty only.
        message: Human-readable explanation.
        cells: Grid coordinates implicated in the violation.
    """

    model_config = ConfigDict(frozen=True)
    rule: str
    severity: Literal["hard", "soft"]
    message: str
    cells: tuple[tuple[int, int], ...] = Field(default_factory=tuple)


class ValidationResult(BaseModel):
    """Outcome of running a level validator.

    Attributes:
        is_valid: Whether the level passes all hard rules.
        violations: Collected rule breaches.
        penalty: Aggregate soft-rule penalty.
    """

    model_config = ConfigDict(frozen=True)
    is_valid: bool
    violations: tuple[Violation, ...] = Field(default_factory=tuple)
    penalty: float = 0.0


class RewardBreakdown(BaseModel):
    """Decomposed reward for a step or terminal transition.

    Attributes:
        total: Scalar reward returned to the learner.
        components: Named partial contributions.
    """

    model_config = ConfigDict(frozen=True)
    total: float
    components: dict[str, float] = Field(default_factory=dict)


class GameplayResult(BaseModel):
    """Aggregated Mario bridge simulation outcome.

    Attributes:
        won: Whether at least one bridge run reached the flag.
        completion_ratio: Mean horizontal progress across runs.
        remaining_time_ms: Time left at episode end (last run).
        mario_mode: Mario power-up state at episode end.
        num_jumps: Total jumps performed.
        max_x_jump: Largest horizontal jump distance.
        max_jump_air_time: Longest air time during a jump.
        kills_total: Total enemies defeated.
        kills_by_type: Per-enemy-type kill counts.
        current_lives: Lives remaining.
        current_coins: Coins collected.
        num_collected_mushrooms: Mushrooms collected.
        num_collected_fireflower: Fire flowers collected.
        num_collected_tile_coins: Coins collected from tile blocks.
        num_destroyed_bricks: Bricks broken.
        num_bump_question_block: Question blocks bumped.
        num_bump_brick: Bricks bumped without breaking.
        num_hurts: Damage events taken.
        game_status: Terminal bridge status string.
        bridge_runs: Number of stochastic runs aggregated.
        bridge_win_rate: Fraction of runs that won.
        bridge_completion_std: Standard deviation of completion ratios.
    """

    model_config = ConfigDict(frozen=True)
    won: bool
    completion_ratio: float
    remaining_time_ms: int
    mario_mode: int
    num_jumps: int
    max_x_jump: float
    max_jump_air_time: int
    kills_total: int
    kills_by_type: dict[str, int]
    current_lives: int
    current_coins: int
    num_collected_mushrooms: int
    num_collected_fireflower: int
    num_collected_tile_coins: int
    num_destroyed_bricks: int
    num_bump_question_block: int
    num_bump_brick: int
    num_hurts: int
    game_status: Literal["WIN", "LOSE", "TIME_OUT", "RUNNING"]
    bridge_runs: int = 1
    bridge_win_rate: float = 0.0
    bridge_completion_std: float = 0.0


class StepState(BaseModel):
    """Transient MDP state during level construction.

    Attributes:
        level: Current partial or complete level grid.
        cursor: Active placement coordinate ``(row, col)``.
        step_count: Number of actions taken so far.
        target: Intent target vector for the episode.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    level: LevelGrid
    cursor: tuple[int, int]
    step_count: int
    target: np.ndarray


class EpisodeState(BaseModel):
    """Terminal MDP state after an episode completes.

    Attributes:
        level: Final level grid.
        target: Intent target vector for the episode.
        validator_result: Structural validation outcome.
        gameplay_result: Optional bridge evaluation outcome.
        step_count: Total actions taken in the episode.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    level: LevelGrid
    target: np.ndarray
    validator_result: ValidationResult
    gameplay_result: GameplayResult | None = None
    step_count: int = 0


class GameplayEvaluator(Protocol):
    """Bridge-backed gameplay evaluator."""

    def evaluate(
        self,
        level: LevelGrid,
        agent: str = "robinBaumgarten",
        *,
        timer_seconds: int = 30,
        mario_state: int = 0,
        num_runs: int = 5,
    ) -> GameplayResult:
        """Run the Mario bridge on a level and aggregate results."""


class RealismScorer(Protocol):
    """Learned or heuristic realism scorer."""

    def score(self, level: LevelGrid) -> float:
        """Return a realism score in ``[0, 1]``."""


class LevelValidator(Protocol):
    """Structural and semantic level validator."""

    def validate(self, level: LevelGrid) -> ValidationResult:
        """Validate a level and return the aggregated result."""


class RewardModel(Protocol):
    """Step and terminal reward model."""

    def step(self, state: StepState, action: int) -> RewardBreakdown:
        """Compute per-step shaping reward."""

    def terminal(self, episode: EpisodeState) -> RewardBreakdown:
        """Compute terminal episode reward."""


class Renderer(Protocol):
    """Level visualisation backend."""

    def render(self, level: LevelGrid) -> Any:
        """Render a level to a backend-specific representation."""


class TrainResult(BaseModel):
    """Outcome of a training run.

    Attributes:
        run_dir: Directory containing run artifacts.
        checkpoint_path: Saved policy checkpoint path.
        total_steps: Environment steps completed.
    """

    model_config = ConfigDict(frozen=True)
    run_dir: Path
    checkpoint_path: Path
    total_steps: int


class PolicyHandle(BaseModel):
    """Loaded policy for generation or evaluation.

    Attributes:
        checkpoint_path: Source checkpoint on disk.
        model: Backend-specific policy object.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    checkpoint_path: Path
    model: Any


class TrainerBackend(Protocol):
    """RL training backend."""

    def train(self, config: Any) -> TrainResult:
        """Train a policy from a runtime configuration."""

    def load_policy(self, checkpoint_path: Path) -> PolicyHandle:
        """Load a trained policy from a checkpoint path."""
