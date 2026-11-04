#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/assets/mario"
TMP="$(mktemp -d)"

trap 'rm -rf "$TMP"' EXIT

git clone --depth 1 https://github.com/amidos2006/Mario-AI-Framework.git "$TMP/mario-ai-framework"
mkdir -p "$DEST"
cp "$TMP/mario-ai-framework/img/"*.png "$TMP/mario-ai-framework/img/"*.gif "$DEST/" 2>/dev/null || true
echo "Mario sprites installed in $DEST"
