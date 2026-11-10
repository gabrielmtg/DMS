#!/usr/bin/env bash
# Gate 1: mede o pipeline na taxa-alvo (10 FPS) e na vazão máxima.
# Uso: bin/benchmark.sh <vídeo ou índice da câmera>
# Durações em segundos: PACED_SECONDS (padrão 300) e MAX_SECONDS (padrão 60).
set -euo pipefail
cd "$(dirname "$0")/.."

source=${1:?uso: bin/benchmark.sh <vídeo ou índice da câmera>}
[[ -x .venv/bin/python ]] || { echo "erro: rode bin/setup_runtime.sh antes" >&2; exit 1; }

stamp=$(hostname)-$(date +%Y%m%d-%H%M%S)
mkdir -p results

echo "== taxa-alvo, 10 FPS (é este resultado que decide o gate)"
.venv/bin/python scripts/benchmark.py --source "$source" --loop --fps 10 \
    --seconds "${PACED_SECONDS:-300}" --out "results/${stamp}_10fps.json"

echo
echo "== vazão máxima"
.venv/bin/python scripts/benchmark.py --source "$source" --loop \
    --seconds "${MAX_SECONDS:-60}" --out "results/${stamp}_max.json"

echo
echo "relatórios em results/${stamp}_*.json"
