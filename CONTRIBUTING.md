# Contributing

Thanks for your interest in improving this artifact. Bug reports, reproducibility issues and
pull requests are welcome.

## Development setup

Requirements: Python 3.11 or 3.12, JDK 17 or newer, GNU Make.

```bash
make setup    # create .venv and install every workspace package with dev extras
make bridge   # build the Java gameplay bridge used by the A* judge
```

`make help` lists every target.

## Before opening a pull request

```bash
make format            # Ruff formatter
make check             # Ruff lint, strict mypy per package, CLI dry-runs, unit tests
make test-integration  # real processed library and Java bridge
```

The same steps run in GitHub Actions on Python 3.11 and 3.12.

## Code layout

Each package under `packages/` has one responsibility; keep new code in the package that owns
the concept:

| Package | Owns |
| --- | --- |
| `pcgrl-core` | contracts, settings, tiles and chunk types (no heavy dependencies) |
| `pcgrl-runtime` | environment, metrics, edits, polish and validators |
| `pcgrl-rendering` | ASCII and PNG rendering |
| `pcgrl-bridge` | Python adapter for the Java gameplay judge |
| `pcgrl-learning` | MaskablePPO training and generation |
| `pcgrl-app` | CLI, pipelines, evaluations and the language bridge |

Tests mirror this layout under `tests/unit/` and `tests/integration/`.

## Results and artifacts

Files under `artifacts/` back the numbers in the paper. A change that alters them should say
which tables move and why, and regenerate them with the matching `make` target.

## Commit style

One-line conventional commits in English, for example:

```text
feat(runtime): add a platform-removal macro-edit
fix(bridge): close the Java process on timeout
docs(readme): document the calibration workflow
```
