#!/usr/bin/env bash
# Do manifesto ao .tflite: crops, treino, exportação e avaliação na validação.
# Uso: bin/train_phone.sh <manifesto.csv> [pasta dos crops, padrão data/phone]
#
# Variáveis opcionais:
#   REBUILD=1          refaz os crops mesmo que já existam (o splits.json é mantido)
#   RUNS=runs/phone    onde ficam os treinos
#   MODEL=models/phone_classifier.tflite
#   TRAIN_ARGS="--head-epochs 1 --finetune-epochs 2"   repassado ao train_phone.py
set -euo pipefail
cd "$(dirname "$0")/.."

manifest=${1:?uso: bin/train_phone.sh <manifesto.csv> [pasta dos crops]}
data=${2:-data/phone}
runs=${RUNS:-runs/phone}
model=${MODEL:-models/phone_classifier.tflite}
py=.venv-train/bin/python
[[ -x $py ]] || { echo "erro: rode bin/setup_train.sh antes" >&2; exit 1; }
read -ra train_args <<< "${TRAIN_ARGS:-}"

echo "== 1/4 crops"
if [[ -f $data/index.csv && ${REBUILD:-0} != 1 ]]; then
    echo "reusando os crops de $data (REBUILD=1 para refazer)"
else
    "$py" scripts/build_phone_dataset.py --manifest "$manifest" --out "$data" --overwrite
fi

echo
echo "== 2/4 treino"
"$py" scripts/train_phone.py --data "$data" --out "$runs" "${train_args[@]}"
# O nome da pasta é data-hora, então a última em ordem alfabética é a que acabou de ser criada.
run=$(find "$runs" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)

echo
echo "== 3/4 exportação"
"$py" scripts/export_tflite.py --run "$run" --out "$model"

echo
echo "== 4/4 avaliação do .tflite na validação"
"$py" scripts/evaluate_phone.py --model "$model" --data "$data" --split val --out "$run/eval_val.json"

echo
echo "modelo: $model (treino em $run)"
