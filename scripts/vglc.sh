#!/usr/bin/env bash
# Download Super Mario Bros levels from the VGLC repository.
#
# Layout:
#   data/raw/corpus/mario/mario-1/W-L.txt   — SMB1 (e.g. 1-1.txt)
#   data/raw/corpus/mario/mario-2/W-L.txt   — SMB2 Japan / Lost Levels
#
# Western SMB2 uses incompatible tile glyphs and is intentionally excluded.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="$ROOT/data/raw/corpus/mario"
TMP="$(mktemp -d)"

trap 'rm -rf "$TMP"' EXIT

echo "Cloning VGLC (shallow)..."
git clone --depth 1 https://github.com/TheVGLC/TheVGLC.git "$TMP/vglc"

mkdir -p "$BASE/mario-1" "$BASE/mario-2"
copied=0
skipped=0

# SMB1 — mario-W-L.txt → mario-1/W-L.txt
SMB1="$TMP/vglc/Super Mario Bros/Processed"
if [[ -d "$SMB1" ]]; then
  for f in "$SMB1"/mario-*.txt; do
    stem="$(basename "$f" .txt)"   # mario-1-1
    # Strip "mario-" prefix to get W-L
    id="${stem#mario-}"            # 1-1
    dest="$BASE/mario-1/${id}.txt"
    if [[ -f "$dest" ]]; then
      skipped=$((skipped + 1))
    else
      cp "$f" "$dest"
      copied=$((copied + 1))
    fi
  done
fi

# SMB2 Japan (The Lost Levels) — SuperMarioBros2(J)-WorldW-L.txt → mario-2/W-L.txt
SMB2J="$TMP/vglc/Super Mario Bros 2 (Japan)/Processed"
if [[ -d "$SMB2J" ]]; then
  for f in "$SMB2J"/SuperMarioBros2\(J\)-World*.txt; do
    [[ -f "$f" ]] || continue
    stem="$(basename "$f" .txt)"
    # SuperMarioBros2(J)-World1-1 → 1-1
    id="$(echo "$stem" | sed 's/SuperMarioBros2(J)-World//')"
    dest="$BASE/mario-2/${id}.txt"
    if [[ -f "$dest" ]]; then
      skipped=$((skipped + 1))
    else
      cp "$f" "$dest"
      copied=$((copied + 1))
    fi
  done
fi

echo "Done: $copied new levels copied, $skipped already present."
echo "SMB1  (mario-1/): $(ls "$BASE/mario-1/"*.txt 2>/dev/null | wc -l) levels"
echo "SMB2J (mario-2/): $(ls "$BASE/mario-2/"*.txt 2>/dev/null | wc -l) levels"
