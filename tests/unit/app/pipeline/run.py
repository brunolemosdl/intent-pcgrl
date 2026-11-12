from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from pcgrl.app.pipeline.run import run_experiment, write_generation_summary


def test_write_generation_summary_records_pass_verdict(tmp_path, config) -> None:
    """Verify write generation summary records pass verdict."""
    level_json = tmp_path / "level.json"
    level_json.write_text(
        json.dumps(
            {
                "status": "SUCCESS_PLAYABLE",
                "diagnostics": {
                    "generation_status": "SUCCESS_PLAYABLE",
                    "structural_valid": True,
                    "heuristic_reachable": True,
                    "terminal_reward": 8.2,
                    "reward_components": {
                        "alignment": 0.9,
                        "style": 0.8,
                        "playability": 1.0,
                        "dup_penalty": 0.0,
                    },
                    "mean_target_error": 0.05,
                    "max_target_error": 0.12,
                    "bridge_checked": True,
                    "bridge_won": True,
                    "bridge_completion_ratio": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )
    summary = write_generation_summary(
        level_json=level_json,
        summary_path=tmp_path / "summary.md",
        config=config,
        training_tier="quick",
    )
    text = summary.read_text(encoding="utf-8")
    assert "Verdict: **PASS**" in text
    assert "Training tier: `quick`" in text


def test_run_experiment_raises_when_corpus_is_missing(tmp_path, config) -> None:
    """Verify run experiment raises when corpus is missing."""
    missing_config = config.model_copy(
        update={"dataset": config.dataset.model_copy(update={"corpus_root": str(tmp_path)})}
    )
    with pytest.raises(FileNotFoundError, match="No corpus levels"):
        run_experiment(missing_config, training_tier="quick")


def test_run_experiment_writes_summary_when_generation_fails(tmp_path, config) -> None:
    """Verify run experiment writes summary when generation fails."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "1-1.txt").write_text("-\n", encoding="utf-8")
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    checkpoint = tmp_path / "policy_final.zip"
    checkpoint.write_bytes(b"fake")
    output_dir = tmp_path / "generated"
    level_json = output_dir / "level_failed.json"
    runtime = config.model_copy(
        update={
            "dataset": config.dataset.model_copy(
                update={
                    "corpus_root": str(corpus_dir),
                    "processed_root": str(processed_dir),
                }
            ),
        }
    )

    def fail_generate(*args, **kwargs) -> Path:
        """Fail generate."""
        output_dir.mkdir(exist_ok=True)
        level_json.write_text(
            json.dumps(
                {
                    "status": "FAIL_NO_PLAYABLE_CANDIDATE",
                    "diagnostics": {
                        "generation_status": "FAIL_NO_PLAYABLE_CANDIDATE",
                        "heuristic_reachable": False,
                    },
                }
            ),
            encoding="utf-8",
        )
        raise RuntimeError("generation failed")

    with (
        patch("pcgrl.app.pipeline.run._processed_corpus_exists", return_value=True),
        patch("pcgrl.app.pipeline.run._chunk_library_exists", return_value=True),
        patch("pcgrl.learning.generate.run_generate", side_effect=fail_generate),
    ):
        result = run_experiment(
            runtime,
            training_tier="quick",
            checkpoint=checkpoint,
            output_dir=output_dir,
        )

    assert result.generation_succeeded is False
    assert result.summary is not None
    assert result.summary.is_file()
