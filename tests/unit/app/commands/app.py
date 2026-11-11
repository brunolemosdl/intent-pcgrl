from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pcgrl.app.commands.app import app
from typer.testing import CliRunner

runner = CliRunner()


def test_prepare_dry_run() -> None:
    """Verify prepare dry run."""
    result = runner.invoke(app, ["prepare", "--dry-run"])
    assert result.exit_code == 0
    assert "Config valid" in result.output


def test_train_dry_run() -> None:
    """Verify train dry run."""
    result = runner.invoke(app, ["train", "--dry-run", "--training", "initial"])
    assert result.exit_code == 0


def test_run_dry_run() -> None:
    """Verify run dry run."""
    result = runner.invoke(app, ["run", "--dry-run", "--training", "quick"])
    assert result.exit_code == 0
    assert "Run config valid" in result.output


def test_train_invalid_tier() -> None:
    """Verify run invalid tier."""
    result = runner.invoke(app, ["train", "--training", "invalid"])
    assert result.exit_code == 1


def test_run_invalid_tier() -> None:
    """Verify generate dry run."""
    result = runner.invoke(app, ["run", "--training", "invalid"])
    assert result.exit_code == 1


def test_generate_dry_run(tmp_path) -> None:
    """Verify evaluate dry run."""
    fake_ckpt = tmp_path / "model.zip"
    fake_ckpt.write_bytes(b"fake")
    result = runner.invoke(app, ["generate", "--checkpoint", str(fake_ckpt), "--dry-run"])
    assert result.exit_code == 0


def test_compare_dry_run() -> None:
    """Verify compare dry run."""
    result = runner.invoke(app, ["compare", "--dry-run"])
    assert result.exit_code == 0


def test_generate_with_target_json_dry_run(tmp_path) -> None:
    """Verify prepare mocked."""
    fake_ckpt = tmp_path / "model.zip"
    fake_ckpt.write_bytes(b"fake")
    target = "[0.3, 0.2, 0.1, 0.3, 0.4, 0.55, 0.6, 0.4]"
    result = runner.invoke(
        app,
        ["generate", "--checkpoint", str(fake_ckpt), "--target-json", target, "--dry-run"],
    )
    assert result.exit_code == 0


def test_prepare_mocked(tmp_path) -> None:
    """Verify train mocked."""
    with patch("pcgrl.app.commands.app.run_prepare", return_value=tmp_path) as mock:
        result = runner.invoke(app, ["prepare"])
        assert result.exit_code == 0
        mock.assert_called_once()
        assert "Prepare complete" in result.output


def test_train_mocked(tmp_path) -> None:
    """Verify generate mocked."""
    with patch("pcgrl.app.commands.app.run_train", return_value=tmp_path / "ckpt.pt"):
        result = runner.invoke(app, ["train", "--training", "initial"])
        assert result.exit_code == 0


def test_train_mocked_with_resume(tmp_path) -> None:
    """Verify train mocked with resume."""
    (tmp_path / "run_dir").mkdir()
    with patch("pcgrl.app.commands.app.run_train", return_value=tmp_path / "ckpt.pt"):
        result = runner.invoke(
            app,
            ["train", "--training", "initial", "--resume", str(tmp_path / "run_dir")],
        )
        assert result.exit_code == 0


def test_generate_mocked(tmp_path) -> None:
    """Verify benchmark mocked."""
    fake_ckpt = tmp_path / "model.zip"
    fake_ckpt.write_bytes(b"fake")
    out_json = tmp_path / "level.json"
    out_json.write_text("{}", encoding="utf-8")
    with patch("pcgrl.app.commands.app.run_generate", return_value=out_json):
        result = runner.invoke(app, ["generate", "--checkpoint", str(fake_ckpt)])
        assert result.exit_code == 0


def test_compare_mocked(tmp_path) -> None:
    """Verify compare mocked."""
    with patch(
        "pcgrl.app.commands.app.run_comparison", return_value=tmp_path / "comparison.csv"
    ) as mock:
        result = runner.invoke(app, ["compare", "--systems", "random_legal,markov"])
        assert result.exit_code == 0
        assert mock.call_args.kwargs["systems"] == ["random_legal", "markov"]


def test_run_mocked(tmp_path) -> None:
    """Verify run mocked."""
    result_value = SimpleNamespace(
        checkpoint=tmp_path / "policy_final.zip",
        level_json=tmp_path / "level.json",
        summary=tmp_path / "summary.md",
        generation_succeeded=True,
    )
    with patch("pcgrl.app.commands.app.run_experiment", return_value=result_value):
        result = runner.invoke(app, ["run", "--training", "quick"])
        assert result.exit_code == 0
        assert "Run complete" in result.output


def test_run_sh_delegates_to_cli() -> None:
    """Verify run sh delegates to cli."""
    script = Path("scripts/run.sh").read_text(encoding="utf-8")
    assert "pcgrl.app.commands run" in script
    assert "MAP_RESULT_SUMMARY" not in script


def test_generate_rejects_prompt_and_target_json(tmp_path) -> None:
    """--prompt and --target-json are mutually exclusive."""
    fake_ckpt = tmp_path / "model.zip"
    fake_ckpt.write_bytes(b"fake")
    result = runner.invoke(
        app,
        [
            "generate",
            "--checkpoint",
            str(fake_ckpt),
            "--target-json",
            "[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]",
            "--prompt",
            "many enemies",
        ],
    )
    assert result.exit_code == 1
    assert "at most one" in result.output


def test_translate_dry_run() -> None:
    """Verify translate dry run."""
    result = runner.invoke(app, ["translate", "--prompt", "many enemies", "--dry-run"])
    assert result.exit_code == 0


def test_translate_with_fake_provider(tmp_path) -> None:
    """The translate command emits a vector fragment with the fake provider."""
    override = tmp_path / "override.yaml"
    override.write_text("translator:\n  provider: fake\n  mode: live\n")
    out_path = tmp_path / "translation.json"
    result = runner.invoke(
        app,
        [
            "translate",
            "--prompt",
            "many enemies, no gaps, flat easy level",
            "--config",
            str(override),
            "--output",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    import json as _json

    payload = _json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["source"] == "fake"
    assert len(payload["vector"]) == 8
    assert payload["vector"][0] >= 0.7  # enemy_density up
    assert payload["vector"][4] <= 0.1  # gap_difficulty down
