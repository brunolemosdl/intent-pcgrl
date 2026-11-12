from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
from rich.console import Console

from pcgrl.core.helpers.log import setup_logging
from pcgrl.core.helpers.path import resolve_path
from pcgrl.core.helpers.seed import seed_all
from pcgrl.core.settings.loader import load_config
from pcgrl.core.settings.schema import RuntimeConfig

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()

DEFAULT_CONFIG = Path("configs/mario.yaml")
TRAINING_TIERS = ("quick", "initial", "medium", "advanced", "final")


def run_prepare(config: RuntimeConfig) -> Path:
    """Run corpus preparation via the pipeline module (lazy import)."""
    from pcgrl.app.pipeline.prepare import run_prepare as pipeline

    return pipeline(config)


def run_library(config: RuntimeConfig) -> Path:
    """Run chunk-library extraction via the pipeline module (lazy import)."""
    from pcgrl.app.pipeline.library import run_library as pipeline

    return pipeline(config)


def run_comparison(
    config: RuntimeConfig,
    *,
    systems: list[str],
    episodes: int = 3,
    checkpoint: Path | None = None,
    with_bridge: bool = False,
    target_source: str = "oracle",
    output_dir: Path | None = None,
) -> Path:
    """Run the system comparison harness via the evaluation module (lazy import)."""
    from pcgrl.app.evaluation.comparison import run_comparison as harness

    return harness(
        config,
        systems=systems,
        episodes=episodes,
        checkpoint=checkpoint,
        with_bridge=with_bridge,
        target_source=target_source,
        output_dir=output_dir,
    )


def run_train(
    config: RuntimeConfig,
    *,
    run_dir: Path | None = None,
    resume: bool = False,
) -> Path:
    """Run policy training via the pipeline module (lazy import)."""
    from pcgrl.app.pipeline.train import run_train as pipeline

    return pipeline(config, run_dir=run_dir, resume=resume)


def run_generate(
    config: RuntimeConfig,
    checkpoint: Path,
    *,
    target: np.ndarray | None = None,
    output_dir: Path | None = None,
    seed: int | None = None,
    translation: dict[str, Any] | None = None,
    polish: bool | None = None,
) -> Path:
    """Run level generation via the pipeline module (lazy import)."""
    from pcgrl.app.pipeline.generate import run_generate as pipeline

    return pipeline(
        config,
        checkpoint,
        target=target,
        output_dir=output_dir,
        seed=seed,
        translation=translation,
        polish=polish,
    )


def run_experiment(
    config: RuntimeConfig,
    *,
    training_tier: str,
    checkpoint: Path | None = None,
    output_dir: Path | None = None,
) -> Any:
    """Run the end-to-end experiment workflow via the pipeline module."""
    from pcgrl.app.pipeline.run import run_experiment as pipeline

    return pipeline(
        config,
        training_tier=training_tier,
        checkpoint=checkpoint,
        output_dir=output_dir,
    )


def _load(
    config: Path,
    *,
    training: str = "medium",
    seed: int | None = None,
) -> RuntimeConfig:
    """Load config, configure logging, and seed RNGs for a CLI command."""
    from pcgrl.core.helpers.env import load_env_file

    load_env_file()
    runtime = load_config(training=training, override_path=config, seed=seed)
    setup_logging(runtime.log.level)
    seed_all(runtime.experiment.seed)
    return runtime


def _validate_training_tier(training: str) -> None:
    """Exit the CLI when ``training`` is not a supported tier name."""
    if training not in TRAINING_TIERS:
        console.print(f"[red]--training must be one of: {', '.join(TRAINING_TIERS)}[/red]")
        raise typer.Exit(1)


@app.command()
def prepare(
    config: Annotated[Path, typer.Option("--config", help="Config path.")] = DEFAULT_CONFIG,
    seed: Annotated[int | None, typer.Option("--seed")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Prepare corpus splits and corpus-derived statistics."""
    runtime = _load(config, seed=seed)
    if dry_run:
        console.print("[green]Config valid[/green]")
        raise typer.Exit(0)
    run_prepare(runtime)
    console.print("[green]Prepare complete[/green]")


@app.command()
def library(
    config: Annotated[Path, typer.Option("--config", help="Config path.")] = DEFAULT_CONFIG,
    seed: Annotated[int | None, typer.Option("--seed")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Extract the annotated chunk library from the raw corpus (composer stage)."""
    runtime = _load(config, seed=seed)
    if dry_run:
        console.print("[green]Config valid[/green]")
        raise typer.Exit(0)
    output_root = run_library(runtime)
    console.print(f"[green]Chunk library saved to {output_root}[/green]")


@app.command()
def compare(
    config: Annotated[Path, typer.Option("--config", help="Config path.")] = DEFAULT_CONFIG,
    systems: Annotated[
        str,
        typer.Option(
            "--systems",
            help=(
                "Comma-separated systems: ppo | ppo_best_of_n | ppo_best_of_n_polished | "
                "ppo_random_edits | ppo_planner_edits | random_legal | greedy_intent | "
                "beam_search | markov | corpus_replay"
            ),
        ),
    ] = "random_legal,greedy_intent,beam_search,markov",
    episodes: Annotated[int, typer.Option("--episodes")] = 3,
    checkpoint: Annotated[
        Path | None, typer.Option("--checkpoint", help="Required for the ppo systems.")
    ] = None,
    with_bridge: Annotated[
        bool, typer.Option("--with-bridge", help="Run the A* bridge per level (slow).")
    ] = False,
    target_source: Annotated[
        str,
        typer.Option(
            "--target-source",
            help="oracle | llm | fake (llm/fake verbalize each case and translate it)",
        ),
    ] = "oracle",
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
    seed: Annotated[int | None, typer.Option("--seed")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Compare systems under one protocol: suite x seeds x episodes (E3/E4)."""
    runtime = _load(config, seed=seed)
    if dry_run:
        console.print("[green]Config valid[/green]")
        raise typer.Exit(0)
    csv_path = run_comparison(
        runtime,
        systems=[name.strip() for name in systems.split(",") if name.strip()],
        episodes=episodes,
        checkpoint=checkpoint,
        with_bridge=with_bridge,
        target_source=target_source,
        output_dir=output_dir,
    )
    console.print(f"[green]Comparison results: {csv_path}[/green]")


@app.command()
def controllability(
    config: Annotated[Path, typer.Option("--config", help="Config path.")] = DEFAULT_CONFIG,
    checkpoint: Annotated[
        Path | None, typer.Option("--checkpoint", help="PPO composer checkpoint.")
    ] = None,
    baseline: Annotated[
        str | None,
        typer.Option("--baseline", help="Evaluate a non-RL baseline instead."),
    ] = None,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
    seed: Annotated[int | None, typer.Option("--seed")] = None,
    polish: Annotated[
        bool,
        typer.Option(
            "--polish",
            help="Sweep the polished product path instead of the raw policy.",
        ),
    ] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Per-dimension controllability sweeps: MAE + Spearman rho (E5)."""
    runtime = _load(config, seed=seed)
    if dry_run:
        console.print("[green]Config valid[/green]")
        raise typer.Exit(0)
    if (checkpoint is None) == (baseline is None):
        console.print("[red]Provide exactly one of --checkpoint or --baseline[/red]")
        raise typer.Exit(1)
    from pcgrl.app.evaluation.controllability import (
        composer_baseline_generate_fn,
        composer_policy_generate_fn,
        composer_policy_polished_generate_fn,
        evaluate_controllability,
        write_controllability_report,
    )

    if checkpoint is not None:
        if polish:
            generate_fn = composer_policy_polished_generate_fn(runtime, checkpoint)
            label = f"{checkpoint.stem}_polished"
        else:
            generate_fn = composer_policy_generate_fn(runtime, checkpoint)
            label = checkpoint.stem
    else:
        generate_fn = composer_baseline_generate_fn(runtime, baseline or "greedy_intent")
        label = baseline or "greedy_intent"
    results = evaluate_controllability(runtime, generate_fn)
    out_dir = output_dir or resolve_path("artifacts/reports/controllability") / label
    report = write_controllability_report(results, out_dir)
    console.print(f"[green]Controllability report: {report}[/green]")


@app.command()
def calibrate(
    config: Annotated[Path, typer.Option("--config", help="Config path.")] = DEFAULT_CONFIG,
    n_levels: Annotated[int | None, typer.Option("--n-levels")] = None,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
    seed: Annotated[int | None, typer.Option("--seed")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Calibrate flood jump physics against the bridge A* (E0; needs the jar)."""
    runtime = _load(config, seed=seed)
    if dry_run:
        console.print("[green]Config valid[/green]")
        raise typer.Exit(0)
    from pcgrl.app.evaluation.calibration import run_calibration

    report = run_calibration(runtime, num_levels=n_levels, output_dir=output_dir)
    console.print(f"[green]Calibration report: {report}[/green]")


@app.command()
def train(
    config: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG,
    training: Annotated[
        str,
        typer.Option(
            "--training", help="Training tier: quick | initial | medium | advanced | final"
        ),
    ] = "medium",
    resume: Annotated[
        str | None,
        typer.Option(
            "--resume", help="Run directory to continue from its latest periodic checkpoint."
        ),
    ] = None,
    seed: Annotated[int | None, typer.Option("--seed")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Train a MaskablePPO composer policy for the selected experiment tier."""
    _validate_training_tier(training)
    runtime = _load(config, training=training, seed=seed)
    if dry_run:
        raise typer.Exit(0)
    run_dir = resolve_path(resume) if resume else None
    checkpoint = run_train(runtime, run_dir=run_dir, resume=run_dir is not None)
    console.print(f"[green]Training checkpoint: {checkpoint}[/green]")


@app.command()
def translate(
    prompt: Annotated[str, typer.Option("--prompt", help="Natural-language description.")],
    output: Annotated[
        Path | None, typer.Option("--output", help="Write the translation JSON here.")
    ] = None,
    config: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Translate a level description into an intent vector (LLM bridge)."""
    runtime = _load(config)
    if dry_run:
        console.print("[green]Config valid[/green]")
        raise typer.Exit(0)
    from pcgrl.app.intent.factory import build_translator

    translation = build_translator(runtime).translate(prompt)
    payload = translation.model_dump(mode="json")
    rendered = json.dumps(payload, indent=2)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
        console.print(f"[green]Translation written to {output}[/green]")
    console.print(rendered)


@app.command()
def language(
    study: Annotated[
        str, typer.Option("--study", help="translation | monotonicity")
    ] = "translation",
    checkpoint: Annotated[
        Path | None, typer.Option("--checkpoint", help="Policy for the monotonicity study.")
    ] = None,
    episodes: Annotated[
        int, typer.Option("--episodes", help="Levels per description (monotonicity).")
    ] = 3,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
    config: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG,
    seed: Annotated[int | None, typer.Option("--seed")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Language-bridge studies: translation error (E7a) or impact monotonicity (E7b)."""
    runtime = _load(config, seed=seed)
    if study not in ("translation", "monotonicity"):
        console.print("[red]--study must be translation or monotonicity[/red]")
        raise typer.Exit(1)
    if dry_run:
        console.print("[green]Config valid[/green]")
        raise typer.Exit(0)
    if study == "monotonicity" and checkpoint is None:
        console.print("[red]--checkpoint is required for the monotonicity study[/red]")
        raise typer.Exit(1)
    from pcgrl.app.experiments.llm_eval import run_monotonicity, run_translation_eval
    from pcgrl.app.intent.factory import build_translator

    translator = build_translator(runtime)
    if study == "translation":
        report = run_translation_eval(runtime, translator, output_dir=output_dir)
    else:
        from pcgrl.app.evaluation.comparison import build_generate_fn

        generate_fn = build_generate_fn(runtime, "ppo_best_of_n", checkpoint=checkpoint)
        report = run_monotonicity(
            runtime, translator, generate_fn, output_dir=output_dir, episodes=episodes
        )
    console.print(f"[green]Language study results: {report}[/green]")


@app.command()
def generate(
    checkpoint: Annotated[Path, typer.Option("--checkpoint")],
    target_json: Annotated[str | None, typer.Option("--target-json")] = None,
    prompt: Annotated[
        str | None,
        typer.Option("--prompt", help="Natural-language description (XOR --target-json)."),
    ] = None,
    seed: Annotated[int | None, typer.Option("--seed")] = None,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
    config: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG,
    polish: Annotated[
        bool | None,
        typer.Option(
            "--polish/--no-polish",
            help="Override generation.polish.enabled for this run.",
        ),
    ] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Generate one level from a trained policy checkpoint."""
    if target_json is not None and prompt is not None:
        console.print("[red]Provide at most one of --target-json or --prompt[/red]")
        raise typer.Exit(1)
    runtime = _load(config, seed=seed)
    if dry_run:
        raise typer.Exit(0)
    translation_record: dict[str, Any] | None = None
    if prompt is not None:
        from pcgrl.app.intent.factory import build_translator

        translation = build_translator(runtime).translate(prompt)
        target = np.asarray(translation.vector, dtype=np.float32)
        translation_record = translation.model_dump(mode="json")
    elif target_json is not None:
        target = np.asarray(json.loads(target_json), dtype=np.float32)
    else:
        target = None
    out = run_generate(
        runtime,
        checkpoint,
        target=target,
        output_dir=output_dir,
        seed=seed,
        translation=translation_record,
        polish=polish,
    )
    console.print(f"[green]Generated {out}[/green]")


@app.command()
def experiment(
    spec: Annotated[Path, typer.Option("--spec", help="Factorial spec YAML path.")],
    stage: Annotated[str, typer.Option("--stage", help="train | evaluate | analyze | all")] = "all",
    checkpoint: Annotated[
        Path | None,
        typer.Option("--checkpoint", help="Shared policy for inference-kind specs."),
    ] = None,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
    config: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG,
    seed: Annotated[int | None, typer.Option("--seed")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Run a replicated factorial experiment from a spec YAML (E6)."""
    from pcgrl.app.experiments.spec import FactorialSpec

    factorial_spec = FactorialSpec.load(spec)
    runtime = _load(config, training=factorial_spec.training, seed=seed)
    if dry_run:
        console.print(
            f"[green]Spec valid: {factorial_spec.name} "
            f"({len(factorial_spec.factors)} factors x r={factorial_spec.replicates})[/green]"
        )
        raise typer.Exit(0)
    from pcgrl.app.experiments.factorial import run_factorial

    out_dir = run_factorial(
        runtime,
        factorial_spec,
        stage=stage,
        output_dir=output_dir,
        checkpoint=checkpoint,
    )
    console.print(f"[green]Experiment artifacts: {out_dir}[/green]")


@app.command()
def run(
    config: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG,
    training: Annotated[
        str,
        typer.Option(
            "--training", help="Training tier: quick | initial | medium | advanced | final"
        ),
    ] = "medium",
    checkpoint: Annotated[Path | None, typer.Option("--checkpoint")] = None,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
    seed: Annotated[int | None, typer.Option("--seed")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Run prepare, library, train, generate, and summary as one workflow."""
    _validate_training_tier(training)
    runtime = _load(config, training=training, seed=seed)
    if dry_run:
        console.print("[green]Run config valid[/green]")
        raise typer.Exit(0)
    result = run_experiment(
        runtime,
        training_tier=training,
        checkpoint=checkpoint,
        output_dir=output_dir,
    )
    status = "generated" if result.generation_succeeded else "diagnostics written"
    console.print(f"[green]Run complete ({status})[/green]")
    console.print(f"[cyan]Checkpoint[/cyan]: {result.checkpoint}")
    if result.level_json is not None:
        console.print(f"[cyan]Level JSON[/cyan]: {result.level_json}")
    if result.summary is not None:
        console.print(f"[cyan]Summary[/cyan]: {result.summary}")


if __name__ == "__main__":
    app()
