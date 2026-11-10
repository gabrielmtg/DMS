# DMS

DMS | Driver Monitoring System

Monitora o motorista em tempo real e alerta sobre sonolência e distração, com alvo na Raspberry Pi 4B: uma única rede de landmarks por quadro e o resto em geometria e janelas temporais.

## Scripts

Os scripts de `bin/` automatizam a instalação e as execuções mais comuns. Rode-os a partir de qualquer pasta.

| Script | O que faz |
|---|---|
| `bin/setup_runtime.sh` | Cria o `.venv` do runtime (notebook ou Pi), baixa o modelo e roda os testes |
| `bin/setup_train.sh` | Cria o `.venv-train` (Python 3.13) para treinar; só no notebook |
| `bin/download_models.sh` | Baixa o modelo do FaceLandmarker e confere o checksum |
| `bin/test.sh` | Roda os testes em cada ambiente que existir; argumentos vão para o pytest |
| `bin/benchmark.sh <vídeo>` | Gate 1: mede na taxa-alvo e na vazão máxima |
| `bin/train_phone.sh <manifesto>` | Do manifesto ao `.tflite`: crops, treino, exportação e avaliação |

Os setups podem rodar de novo para atualizar as dependências.

## Instalação

```bash
bin/setup_runtime.sh
```

Na Raspberry Pi, use o **Raspberry Pi OS 64-bit**, porque o MediaPipe não tem wheels para armv7; o script recusa sistemas 32-bit. Se `cv2` não importar por falta de `libGL`, o script mostra o pacote do `apt` a instalar.

## Gate 1: desempenho na Raspberry Pi

Use um vídeo com uma pessoa olhando para a câmera, como no carro, e rode na própria Pi, com dissipador:

```bash
bin/benchmark.sh video.mp4
```

O script mede 5 minutos a 10 FPS, que é o resultado que decide o gate, e 1 minuto em vazão máxima, que mostra a folga. Os relatórios ficam em `results/`. Para mudar as durações, use `PACED_SECONDS` e `MAX_SECONDS`, em segundos.

Confira `face` no relatório: abaixo de 100%, parte dos quadros mediu só o detector, que é mais barato, e o resultado fica otimista. As metas estão em `configs/default.yaml`.

## Modelo de celular

MobileNetV3-Small sobre um crop do rosto, com 4 classes: `normal`, `phone_call`, `drinking` e `hair_makeup`. É treinado em PyTorch e exportado em `.tflite` float32 pelo `litert-torch`.

### Ambiente de treino

Fica só no notebook, separado do runtime e com **Python 3.13**. O `litert-torch` acaba exigindo uma versão antiga do `pydantic`, que não tem wheel para o 3.14.

```bash
bin/setup_train.sh
```

O script usa o `python3.13` do sistema ou, se não houver, cria o ambiente com o conda. O torch é instalado pelo índice CPU antes do resto, senão o pip baixaria as versões com CUDA.

### Do manifesto ao `.tflite`

```bash
bin/train_phone.sh data/dmd/manifest.csv
```

- **Crops:** se já existirem em `data/phone`, são reaproveitados; use `REBUILD=1` para refazer.
- **Modelo:** cada execução sobrescreve `models/phone_classifier.tflite`. Use `MODEL=...` para gravar em outro lugar.
- **Avaliação:** o resultado da validação fica em `runs/phone/<data-hora>/eval_val.json`.
- **Opções do treino:** passe em `TRAIN_ARGS`, por exemplo `TRAIN_ARGS="--finetune-epochs 20"`.

Para rodar uma etapa só:

```bash
.venv-train/bin/python scripts/build_phone_dataset.py --manifest data/dmd/manifest.csv --out data/phone
.venv-train/bin/python scripts/train_phone.py --data data/phone
.venv-train/bin/python scripts/export_tflite.py --run runs/phone/<data-hora>
.venv-train/bin/python scripts/evaluate_phone.py --data data/phone --split val
```

- **Build:** mostra, por classe, quantos quadros ficaram sem rosto. Se a taxa for alta em `phone_call`, a mão está escondendo o rosto, e o runtime vai sofrer do mesmo jeito.
- **Split:** é por motorista e fica salvo em `data/phone/splits.json`. Use o mesmo arquivo em todos os modelos; o conjunto de teste é para a avaliação final, uma vez só.
- **Exportação:** só grava o modelo se o `.tflite` passar nas verificações (formato, ops nativas e mesma saída que o PyTorch).

### Entrada: manifesto

CSV com as colunas `video,frame,label,subject`, uma linha por quadro:

| Coluna | Conteúdo |
|---|---|
| `video` | caminho do vídeo RGB da câmera de rosto |
| `frame` | índice do quadro nesse vídeo, a partir de 0 |
| `label` | rótulo do nível 6 do DMD ("Driver actions"), escrito como no JSON |
| `subject` | id do motorista |

Pode mandar todos os quadros: o build usa 1 a cada 10. Rótulos que não estão em `configs/phone.yaml` param o build com a lista do que faltou mapear.

### Saída: modelo

Cada `.tflite` sai acompanhado de um `.json` com o mesmo nome, contendo as classes na ordem da saída, a especificação da entrada e os parâmetros do crop. Leia as classes desse arquivo em vez de fixá-las no código.

- **Entrada:** `[1,160,160,3]`, float32, NHWC, RGB, de 0 a 255. A normalização fica dentro do grafo.
- **Saída:** `[1,4]` float32, já com softmax.

O crop é a parte que precisa sair idêntica no C++. A referência é [src/dms/crop.py](src/dms/crop.py):

1. Multiplique os 478 landmarks normalizados pela largura e altura da imagem, **em double**.
2. `s` = maior lado da caixa dos pontos; `cx, cy` = centro da caixa.
3. `left = floor(cx - scale*s/2 + 0.5)`, `top = floor(cy + shift*s - scale*s/2 + 0.5)` e `side = floor(scale*s + 0.5)`.
4. A parte do quadrado fora da imagem fica preta (0).
5. Redimensione para 160×160 com `INTER_AREA`.

Use `floor(x + 0.5)`, não `std::round` nem `cvRound`: os três discordam quando o valor termina em .5, inclusive em números negativos.

`tests/golden/crop/` tem casos para comparar: `input.png` e, por caso em `cases.json`, os landmarks, a caixa esperada e o crop esperado (`crop_<nome>.png`). A caixa precisa bater exatamente. Os pixels podem diferir em até 1 entre versões do OpenCV. Se `src/dms/crop.py` mudar, gere os casos de novo com `scripts/export_crop_golden.py`.

Dados (`data/`) e modelos (`models/`) ficam fora do git. A licença do DMD (CC BY-NC-ND 4.0) não permite redistribuição.
