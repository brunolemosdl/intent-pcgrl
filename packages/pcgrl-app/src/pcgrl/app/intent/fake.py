from __future__ import annotations

import re
from datetime import UTC, datetime

from pcgrl.app.intent.prompt import PROMPT_VERSION
from pcgrl.app.intent.types import IntentTranslation, clamp_vector
from pcgrl.core.settings.schema import RuntimeConfig

INTENSITY_BANDS: tuple[tuple[float, tuple[str, ...]], ...] = (
    (0.05, ("no", "none", "zero", "without", "free of")),
    (0.20, ("few", "rare", "sparse", "barely", "little", "occasional")),
    (0.40, ("some", "moderate", "a bit of", "several")),
    (0.70, ("many", "lots of", "plenty of", "frequent", "abundant", "loads of")),
    (0.95, ("maximum", "max", "extreme", "as many", "packed with", "tons of")),
)

DIMENSION_SYNONYMS: dict[str, tuple[str, ...]] = {
    "enemy_density": ("enemies", "enemy", "goombas", "goomba", "monsters", "monster"),
    "coin_density": ("coins", "coin"),
    "interactive_block_density": (
        "blocks",
        "block",
        "bricks",
        "brick",
        "question marks",
        "question blocks",
    ),
    "pipe_density": ("pipes", "pipe", "tubes", "tube"),
    "gap_difficulty": ("gaps", "gap", "pits", "pit", "holes", "hole", "chasms"),
    "verticality": ("vertical", "verticality", "hills", "hilly", "climbing", "climbs"),
    "path_smoothness": ("smoothness", "even path", "gentle path"),
    "platform_usage": ("platforms", "platform", "floating platforms"),
}

# Adjectives that set dimensions directly, outside the intensity x synonym
# grid (e.g. "flat" pins verticality low AND smoothness high).
ADJECTIVE_RULES: tuple[tuple[str, dict[str, float]], ...] = (
    ("flat", {"verticality": 0.05, "path_smoothness": 0.95}),
    ("hilly", {"verticality": 0.70}),
    ("mountainous", {"verticality": 0.95}),
    ("bumpy", {"path_smoothness": 0.20}),
    ("rough", {"path_smoothness": 0.20}),
    ("smooth", {"path_smoothness": 0.95}),
    ("easy", {"gap_difficulty": 0.20, "enemy_density": 0.20}),
    ("hard", {"gap_difficulty": 0.80, "enemy_density": 0.70}),
    ("difficult", {"gap_difficulty": 0.80, "enemy_density": 0.70}),
)

_MENTIONED_DEFAULT = 0.70


class FakeIntentTranslator:
    """Deterministic keyword-lexicon translator for CI and the rule-based arm.

    Description:
        Splits the description into clauses, looks for dimension synonyms, and
        assigns the band of the nearest intensity word in the same clause
        (mention without intensity reads as "many"). Unmentioned dimensions
        fall back to the config defaults — the same contract the grounded
        prompt instructs the real LLM to follow, which is what makes the
        ``fake(verbalize(v))`` round-trip property testable.
    """

    def __init__(self, config: RuntimeConfig) -> None:
        """Bind the config defaults used for unmentioned dimensions."""
        self._names = config.target_vector_field_names()
        self._defaults = config.default_target_vector()

    def translate(self, description: str) -> IntentTranslation:
        """Translate one description through the keyword lexicon.

        Args:
            description: Natural-language level description.

        Returns:
            Deterministic translation record (source=``fake``).

        Description:
            Explicit dimension mentions ("many enemies") outrank adjective
            implications ("easy"), so a description combining both keeps the
            explicit value.
        """
        text = description.lower()
        explicit: dict[str, float] = {}
        implied: dict[str, float] = {}
        for clause in re.split(r"[,;.]| and | but ", text):
            self._apply_clause(clause.strip(), explicit, implied)
        values = dict(zip(self._names, self._defaults, strict=True))
        values.update(implied)
        values.update(explicit)
        vector, clamped = clamp_vector(tuple(values[name] for name in self._names))
        return IntentTranslation(
            description=description,
            vector=vector,
            source="fake",
            model=None,
            prompt_version=PROMPT_VERSION,
            clamped=clamped,
            rationale=None,
            created_at=datetime.now(tz=UTC).isoformat(),
        )

    def _apply_clause(
        self, clause: str, explicit: dict[str, float], implied: dict[str, float]
    ) -> None:
        """Collect adjective implications and intensity x synonym matches."""
        if not clause:
            return
        for adjective, updates in ADJECTIVE_RULES:
            if re.search(rf"\b{re.escape(adjective)}\b", clause):
                implied.update(updates)
        intensity = self._clause_intensity(clause)
        for dim, synonyms in DIMENSION_SYNONYMS.items():
            if any(re.search(rf"\b{re.escape(s)}\b", clause) for s in synonyms):
                explicit[dim] = (
                    intensity if intensity is not None else _MENTIONED_DEFAULT
                )

    @staticmethod
    def _clause_intensity(clause: str) -> float | None:
        """Return the band of the strongest intensity phrase in the clause.

        Bands are checked from the extreme end down so "as many as possible"
        resolves to the maximum band before the bare "many" can match.
        """
        for value, words in reversed(INTENSITY_BANDS):
            if any(re.search(rf"\b{re.escape(w)}\b", clause) for w in words):
                return value
        return None
