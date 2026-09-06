#!/usr/bin/env bash
# Launch one D&R training run on the 4x A800 devbox (paths per myfile/WORK_MEMORY.md, 2026-09-05).
#   usage: bash scripts/launch_dnr_devbox.sh <EXP_NAME> [HISTORY_YAML] [NUM_STEPS]
#   e.g. : bash scripts/launch_dnr_devbox.sh dnr_bud64_pool128 perceptual-dnr-modul_bud64_pool128.yaml 40000
# Override any path with an env var (DNR_ROOT, VENV_PY, DATA, OPENPI_HOME, WANDB_OVERLAY, PROXY, GPUS,
# MEM_FRAC, DATASET_TYPE). Runs from the dnr checkout with its own src on PYTHONPATH, reusing the a2r venv
# (identical deps) -- same trick as the local eval. Refuses to launch on top of another train.py or an
# existing checkpoint dir (no accidental --overwrite).
set -euo pipefail
EXP=${1:?exp name}; YAML=${2:-perceptual-dnr-modul_bud64_pool128.yaml}; STEPS=${3:-40000}
DNR_ROOT=${DNR_ROOT:-/workspace/mnt/mywang87/0Xuehui/robomme_policy_learning_dnr}
VENV_PY=${VENV_PY:-/workspace/mnt/mywang87/0Xuehui/robomme_a2r/.venv/bin/python}
DATA=${DATA:-/workspace/mnt/mywang87/0Xuehui/robomme_a2r/data/robomme_preprocessed_data}
OPENPI_HOME=${OPENPI_HOME:-/workspace/mnt/mywang87/.cache/openpi}
WANDB_OVERLAY=${WANDB_OVERLAY:-/workspace/mnt/mywang87/0Xuehui/wandb_overlay}
PROXY=${PROXY:-http://10.2.83.188:3128}
GPUS=${GPUS:-0,1,2,3}; MEM_FRAC=${MEM_FRAC:-0.85}; DATASET_TYPE=${DATASET_TYPE:-auto}

# dataset format: bin reads <DATA>/features_bin/image_emb_*/<episode>.bin (convert_features_to_bin.py); npy reads <DATA>/features
if [ "$DATASET_TYPE" = auto ]; then
  if ls "$DATA"/features_bin/image_emb_4x4/*.bin >/dev/null 2>&1; then DATASET_TYPE=bin; else DATASET_TYPE=npy; fi
  echo "dataset type auto-detected: $DATASET_TYPE   ($(ls "$DATA" | head -6 | tr "\n" " "))"
fi

cd "$DNR_ROOT"
[ -f "src/mme_vla_suite/models/config/robomme/$YAML" ] || { echo "missing history yaml: $YAML"; exit 1; }
[ -d "$DATA" ] || { echo "missing dataset dir: $DATA"; exit 1; }
[ -x "$VENV_PY" ] || { echo "missing venv python: $VENV_PY"; exit 1; }
[ -d "$WANDB_OVERLAY" ] || { echo "missing wandb overlay: $WANDB_OVERLAY (needed for W&B 0.29 / 86-char key)"; exit 1; }
if pgrep -f '[s]cripts/train.py' >/dev/null; then
  echo "another scripts/train.py is alive -- check before launching:"; pgrep -af '[s]cripts/train.py'; exit 1; fi
if [ -d "runs/ckpts/mme_vla_suite/$EXP" ]; then
  echo "runs/ckpts/mme_vla_suite/$EXP already exists -- choose a new EXP_NAME or resume by hand"; exit 1; fi
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

LOG=$DNR_ROOT/${EXP}.log; PIDF=/tmp/${EXP}.pid
PP=$WANDB_OVERLAY:$DNR_ROOT/src:$DNR_ROOT/packages/openpi-client/src
nohup setsid -f /bin/bash -c "echo \$\$ > $PIDF; exec env CUDA_VISIBLE_DEVICES=$GPUS XLA_PYTHON_CLIENT_MEM_FRACTION=$MEM_FRAC \
  OPENPI_DATA_HOME=$OPENPI_HOME http_proxy=$PROXY https_proxy=$PROXY HTTP_PROXY=$PROXY HTTPS_PROXY=$PROXY NO_PROXY=localhost,127.0.0.1 \
  PYTHONPATH=$PP WANDB_X_SERVICE_WAIT=120 WANDB_INIT_TIMEOUT=300 \
  $VENV_PY scripts/train.py mme_vla_suite --exp-name=$EXP --batch-size=64 --num-workers=4 --fsdp-devices=4 \
  --dataset-path=$DATA --dataset-type=$DATASET_TYPE --num-read-threads=1 \
  --model.use-history --model.history-config=$YAML --num-train-steps=$STEPS --save-interval=10000 \
  --wandb-enabled --log-interval=10" >> "$LOG" 2>&1 < /dev/null &
echo "launched $EXP  log=$LOG  pidfile=$PIDF"
echo "watch:  tail -n 200 $LOG | grep -E 'Step [0-9]+:|keep_frac|Traceback|out of memory' | tail"
