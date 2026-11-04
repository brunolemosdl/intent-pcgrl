#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/vendor/mario"
mkdir -p build lib fatjar
if [[ ! -f lib/json.jar ]]; then
  curl -fsSL -o lib/json.jar https://repo1.maven.org/maven2/org/json/json/20240303/json-20240303.jar
fi
find source -name '*.java' > sources.txt
javac -encoding UTF-8 -cp lib/json.jar -d build @sources.txt
rm -rf fatjar && mkdir fatjar
(
  cd fatjar
  jar xf ../lib/json.jar
  cp -r ../build/* .
  jar cfe ../mario-bridge.jar org.intentpcgrl.bridge.Bridge .
)
echo "Built vendor/mario/mario-bridge.jar"
