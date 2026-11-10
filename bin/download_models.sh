#!/usr/bin/env bash
# Baixa o modelo do FaceLandmarker para models/ e confere o checksum.
set -euo pipefail
cd "$(dirname "$0")/.."

URL=https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
SHA256=64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff
OUT=models/face_landmarker.task

check() {
    echo "$SHA256  $1" | sha256sum --check --status
}

if [[ -f $OUT ]]; then
    if check "$OUT"; then
        echo "modelo ok: $OUT"
        exit 0
    fi
    echo "erro: $OUT existe mas o checksum não bate; apague o arquivo e rode de novo" >&2
    exit 1
fi

mkdir -p models
curl -fL --retry 3 -o "$OUT.tmp" "$URL"
if ! check "$OUT.tmp"; then
    rm -f "$OUT.tmp"
    echo "erro: o arquivo baixado não tem o checksum esperado" >&2
    exit 1
fi
mv "$OUT.tmp" "$OUT"
echo "modelo baixado: $OUT"
