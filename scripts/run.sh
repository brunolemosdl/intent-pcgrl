#!/usr/bin/env bash
# End-to-end experiment runner: prepare -> library -> train -> generate -> summary.
#
# Usage:
#   scripts/run.sh [tier] [extra pcgrl-run flags...]
#   TRAINING=quick CONFIG=configs/mario.yaml scripts/run.sh
#
# Tiers: quick | initial | medium | advanced | final (default: medium)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TIERS=(quick initial medium advanced final)
TRAINING="${TRAINING:-medium}"
CONFIG="${CONFIG:-configs/mario.yaml}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"

usage() {
  echo "Usage: scripts/run.sh [tier] [extra flags...]"
  echo "Tiers: ${TIERS[*]} (default: medium)"
  echo "Env overrides: TRAINING, CONFIG, PYTHON"
}

if [[ $# -gt 0 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
  usage
  exit 0
fi

if [[ $# -gt 0 && "$1" != --* ]]; then
  TRAINING="$1"
  shift
fi

if [[ ! " ${TIERS[*]} " == *" ${TRAINING} "* ]]; then
  echo "error: unknown training tier '${TRAINING}'" >&2
  usage >&2
  exit 2
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "error: python interpreter not found at $PYTHON (run 'make setup' first)" >&2
  exit 2
fi

if [[ ! -f "$ROOT/vendor/mario/mario-bridge.jar" ]]; then
  echo "==> Mario bridge JAR missing; building it"
  bash "$ROOT/scripts/bridge.sh"
fi

if [[ ! -d "$ROOT/data/raw/corpus/mario" ]]; then
  echo "==> VGLC corpus missing; downloading it"
  bash "$ROOT/scripts/vglc.sh"
fi

echo "==> Running full pipeline (tier=${TRAINING}, config=${CONFIG})"
exec "$PYTHON" -m pcgrl.app.commands run \
  --training "$TRAINING" \
  --config "$CONFIG" \
  "$@"
