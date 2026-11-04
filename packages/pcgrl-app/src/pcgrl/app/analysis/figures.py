from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _matplotlib() -> Any:
    """Import matplotlib lazily; the ``analysis`` extra is optional."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        logger.info("matplotlib not installed; skipping figure (pip install '.[analysis]')")
        return None


def effects_pareto(effects: list[dict[str, object]], path: Path) -> Path | None:
    """Write a Pareto bar chart of factorial effect magnitudes.

    Args:
        effects: Ranked output of ``factorial_effects``.
        path: Destination PNG path.

    Returns:
        The written path, or ``None`` when matplotlib is unavailable.
    """
    plt = _matplotlib()
    if plt is None or not effects:
        return None
    names = [str(item["name"]) for item in effects]
    magnitudes = [abs(float(item["effect"])) for item in effects]  # type: ignore[arg-type]
    significant = [float(item["p_value"]) < 0.05 for item in effects]  # type: ignore[arg-type]
    colors = ["tab:blue" if sig else "tab:gray" for sig in significant]
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.4 * len(names))))
    ax.barh(range(len(names)), magnitudes, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("|effect| (blue: p < 0.05)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def comparison_box(
    values_by_system: dict[str, list[float]], metric: str, path: Path
) -> Path | None:
    """Write a box plot of one metric across systems.

    Args:
        values_by_system: Metric values keyed by system name.
        metric: Metric label for the y-axis.
        path: Destination PNG path.

    Returns:
        The written path, or ``None`` when matplotlib is unavailable.
    """
    plt = _matplotlib()
    if plt is None or not values_by_system:
        return None
    names = list(values_by_system)
    fig, ax = plt.subplots(figsize=(max(4, 1.2 * len(names)), 4))
    ax.boxplot([values_by_system[name] for name in names], tick_labels=names)
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
