.DEFAULT_GOAL := help
SHELL := /bin/bash

VENV := .venv
PYTHON ?= $(VENV)/bin/python
PIP ?= $(VENV)/bin/pip
RUFF ?= $(VENV)/bin/ruff
MYPY ?= $(VENV)/bin/mypy
CLI := $(PYTHON) -m pcgrl.app.commands

CONFIG ?= configs/mario.yaml
TRAINING ?= medium
CHECKPOINT ?=
BASELINE ?=
SYSTEMS ?= random_legal,greedy_intent,beam_search,markov
EPISODES ?= 3
SPEC ?= configs/experiments/factorial_composer.yaml
STAGE ?= all
STUDY ?= translation
N_LEVELS ?= 120
MPLCONFIGDIR ?= $(CURDIR)/.cache/matplotlib
BRIDGE_JAR := vendor/mario/mario-bridge.jar

export MPLCONFIGDIR

PACKAGES := pcgrl-core pcgrl-runtime pcgrl-rendering pcgrl-bridge pcgrl-learning pcgrl-app
PACKAGE_DIRS := $(addprefix packages/,$(PACKAGES))

.PHONY: help setup install assets bridge ensure-bridge vglc \
        prepare library train run generate compare experiment \
        controllability calibrate language \
        lint format typecheck check command-dry-runs test test-unit \
        test-integration coverage clean clean-bridge clean-artifacts

##@ Setup

help: ## Show this help.
	@awk 'BEGIN {FS = ":.*?## "} \
		/^##@/ {printf "\n\033[1m%s\033[0m\n", substr($$0, 5)} \
		/^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Create the virtualenv and install all workspace packages, optional extras and dev tools.
	python3 -m venv $(VENV)
	$(PIP) install -U pip wheel
	@for pkg in $(PACKAGES); do $(PIP) install -e packages/$$pkg || exit 1; done
	$(PIP) install -e "packages/pcgrl-app[llm,analysis]"
	$(PIP) install -e ".[dev]"

install: setup ## Alias for setup.

assets: ## Download Mario sprite sheets into assets/mario/.
	bash scripts/assets.sh

bridge: ## Build the Mario gameplay bridge JAR.
	bash scripts/bridge.sh

ensure-bridge: ## Build the Mario bridge only when the JAR is missing.
	@if [ ! -f "$(BRIDGE_JAR)" ]; then $(MAKE) bridge; fi

vglc: ## Download VGLC SMB levels into data/raw/corpus/mario/.
	bash scripts/vglc.sh

##@ Pipeline

prepare: ## Build corpus segments, profile, and TPKL pattern distributions.
	$(CLI) prepare --config $(CONFIG)

library: ## Extract the annotated chunk library and the intent viability map.
	$(CLI) library --config $(CONFIG)

train: ## Train the composer policy. Override TRAINING=quick|initial|medium|advanced|final.
	$(CLI) train --training $(TRAINING) --config $(CONFIG)

run: ensure-bridge ## Full pipeline: prepare, library, train, generate, summary.
	bash scripts/run.sh $(TRAINING) --config $(CONFIG)

generate: ensure-bridge ## Generate a level from CHECKPOINT=path.
	@test -n "$(CHECKPOINT)" || (echo "CHECKPOINT is required" >&2; exit 2)
	$(CLI) generate --checkpoint $(CHECKPOINT) --config $(CONFIG)

##@ Evaluation

compare: ## Compare systems (E3/E4). Override SYSTEMS=a,b EPISODES=n CHECKPOINT=path.
	@if [ -n "$(CHECKPOINT)" ]; then \
		$(CLI) compare --systems $(SYSTEMS) --episodes $(EPISODES) --checkpoint $(CHECKPOINT) --config $(CONFIG); \
	else \
		$(CLI) compare --systems $(SYSTEMS) --episodes $(EPISODES) --config $(CONFIG); \
	fi

controllability: ## Per-dimension sweeps (E5) with CHECKPOINT=path or BASELINE=name.
	@if [ -n "$(CHECKPOINT)" ]; then \
		$(CLI) controllability --checkpoint $(CHECKPOINT) --config $(CONFIG); \
	elif [ -n "$(BASELINE)" ]; then \
		$(CLI) controllability --baseline $(BASELINE) --config $(CONFIG); \
	else \
		echo "CHECKPOINT or BASELINE is required" >&2; exit 2; \
	fi

calibrate: ensure-bridge ## Calibrate flood jump physics against the bridge A* (E0).
	$(CLI) calibrate --n-levels $(N_LEVELS) --config $(CONFIG)

language: ## Language-bridge studies. Override STUDY=translation|monotonicity CHECKPOINT=path.
	@if [ -n "$(CHECKPOINT)" ]; then \
		$(CLI) language --study $(STUDY) --checkpoint $(CHECKPOINT) --config $(CONFIG); \
	else \
		$(CLI) language --study $(STUDY) --config $(CONFIG); \
	fi

experiment: ## Run a factorial experiment. Override SPEC=path STAGE=train|evaluate|analyze|all.
	$(CLI) experiment --spec $(SPEC) --stage $(STAGE) --config $(CONFIG)

##@ Quality

lint: ## Run ruff checks.
	$(RUFF) check packages tests main.py

format: ## Format Python sources.
	$(RUFF) format packages tests main.py

typecheck: ## Run mypy per workspace package to avoid namespace package collisions.
	@for pkg in $(PACKAGE_DIRS); do $(MYPY) --explicit-package-bases $$pkg/src || exit 1; done

command-dry-runs: ## Validate key CLI commands without running training.
	$(CLI) prepare --config $(CONFIG) --dry-run
	$(CLI) library --config $(CONFIG) --dry-run
	$(CLI) train --config $(CONFIG) --training quick --dry-run
	$(CLI) compare --config $(CONFIG) --dry-run
	$(CLI) calibrate --config $(CONFIG) --dry-run
	$(CLI) language --config $(CONFIG) --dry-run
	$(CLI) run --config $(CONFIG) --training quick --dry-run

check: lint typecheck command-dry-runs test-unit ## Run local verification.

test: test-unit test-integration ## Run all tests.

test-unit: ## Run all unit tests.
	$(PYTHON) -m pytest tests/unit

test-integration: ## Run integration tests.
	$(PYTHON) -m pytest tests/integration

coverage: ## Run tests with coverage across the pcgrl namespace.
	$(PYTHON) -m pytest --cov=pcgrl --cov-report=term-missing

##@ Cleanup

clean: ## Remove caches and build scraps ONLY (never artifacts/, data/, or the bridge JAR).
	rm -rf build dist .eggs htmlcov .generate_tmp
	rm -rf .pytest_cache .mypy_cache .ruff_cache .cache
	rm -f .coverage .coverage.* coverage.json coverage.xml
	rm -f map.png map.json map.ascii MAP_RESULT_SUMMARY.md
	rm -f level_*.png level_*.json level_*.ascii
	find packages tests -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find packages -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	find packages tests -type f -name '*.pyc' -delete 2>/dev/null || true

clean-bridge: ## Remove the built bridge JAR and Java build scratch (rebuild with `make bridge`).
	rm -rf vendor/mario/build vendor/mario/fatjar vendor/mario/target
	rm -f vendor/mario/sources.txt vendor/mario/mario-bridge.jar

clean-artifacts: ## DESTRUCTIVE: delete artifacts/ (checkpoints, reports, experiments) and data/processed/. Requires FORCE=1.
	@if [ "$(FORCE)" != "1" ]; then \
		echo "Refusing: this would delete artifacts/ (trained checkpoints, reports,"; \
		echo "experiment fragments) and data/processed/ (chunk library, viability map)."; \
		echo "If you really mean it: make clean-artifacts FORCE=1"; \
		exit 2; \
	fi
	rm -rf artifacts data/processed
