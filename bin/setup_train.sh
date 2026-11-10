#!/usr/bin/env bash
# Cria o .venv-train (Python 3.13) para treinar o modelo de celular. Só no notebook.
# O litert-torch acaba exigindo um pydantic antigo, sem wheel para o Python 3.14.
set -euo pipefail
cd "$(dirname "$0")/.."

ENV=.venv-train
TORCH_INDEX=https://download.pytorch.org/whl/cpu

die() {
    echo "erro: $*" >&2
    exit 1
}

find_conda() {
    # Em script, o `conda` do shell interativo (uma função) não existe; procura o executável.
    local candidate
    for candidate in "${CONDA_EXE:-}" "$(command -v conda 2>/dev/null || true)" \
        "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda"; do
        if [[ -n $candidate && -x $candidate ]]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

if [[ ! -x $ENV/bin/python ]]; then
    if command -v python3.13 >/dev/null; then
        python3.13 -m venv "$ENV"
    elif conda=$(find_conda); then
        "$conda" create -y -p "$ENV" -c conda-forge --override-channels python=3.13 pip
    else
        die "precisa de python3.13 ou do conda para criar o $ENV"
    fi
fi
"$ENV/bin/python" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 13))' \
    || die "$ENV não está em Python 3.13; apague a pasta e rode de novo"

"$ENV/bin/pip" install --upgrade pip
# O torch vem antes e pelo índice CPU; pelo PyPI o pip baixaria GBs de CUDA.
"$ENV/bin/pip" install --index-url "$TORCH_INDEX" 'torch>=2.14,<2.15' torchvision
"$ENV/bin/pip" install -e '.[train,dev]'

bin/download_models.sh
"$ENV/bin/python" -m pytest -q
echo
echo "treino pronto em $ENV"
