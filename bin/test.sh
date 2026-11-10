#!/usr/bin/env bash
# Roda os testes em cada ambiente que existir (.venv e .venv-train).
# Argumentos vão para o pytest, ex.: bin/test.sh -k crop
set -uo pipefail
cd "$(dirname "$0")/.."

status=0
found=0
for env in .venv .venv-train; do
    [[ -x $env/bin/python ]] || continue
    found=1
    echo "== $env"
    "$env/bin/python" -m pytest -q "$@" || status=1
done
if (( !found )); then
    echo "erro: nenhum ambiente encontrado; rode bin/setup_runtime.sh" >&2
    exit 1
fi
exit $status
