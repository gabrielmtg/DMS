#!/usr/bin/env bash
# Cria o .venv do runtime (notebook ou Raspberry Pi), instala as dependências,
# baixa o modelo e roda os testes. Pode rodar de novo para atualizar.
# Outro Python: PYTHON=python3.12 bin/setup_runtime.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python3}

die() {
    echo "erro: $*" >&2
    exit 1
}

arch=$(uname -m)
if [[ $arch == armv7l || $arch == armv6l ]]; then
    die "sistema 32-bit ($arch). Use o Raspberry Pi OS 64-bit: o MediaPipe não tem wheels para armv7."
fi
command -v "$PYTHON" >/dev/null || die "não encontrei $PYTHON"
"$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 11))' \
    || die "precisa de Python 3.11 ou mais novo (achei $("$PYTHON" --version))"

if [[ ! -x .venv/bin/python ]]; then
    "$PYTHON" -m venv .venv || die "não consegui criar o .venv; no Raspberry Pi OS: sudo apt install python3-venv"
fi
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e '.[dev]'

if ! .venv/bin/python -c 'import cv2, mediapipe' 2>/dev/null; then
    .venv/bin/python -c 'import cv2, mediapipe' || true
    die "cv2 ou mediapipe não importam. Se o erro acima falar de libGL ou libgthread: sudo apt install libgl1 libglib2.0-0"
fi

bin/download_models.sh
.venv/bin/python -m pytest -q
echo
echo "runtime pronto em .venv"
