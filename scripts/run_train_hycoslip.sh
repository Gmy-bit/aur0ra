#!/usr/bin/env bash
# AURORA ViT-S/16 launcher. This wrapper supplies the model/data/regularization
# defaults used by the official training recipe, then delegates to the shared
# configurable launcher.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# This model-factory alias selects the MERU-family implementation used by the
# AURORA route. `hycoclip` is a different compatibility model and is not used.
MODEL_NAME="${MODEL_NAME:-hycoslip}"
if [[ "${MODEL_NAME}" != "hycoslip" ]]; then
  echo "[ERROR] run_train_hycoslip.sh requires MODEL_NAME=hycoslip (got: ${MODEL_NAME})." >&2
  echo "        Use run_train_sigmeru.sh for other model families." >&2
  exit 2
fi

USE_BOXES="${USE_BOXES:-true}"
if [[ "${USE_BOXES}" != "true" ]]; then
  echo "[ERROR] AURORA training requires USE_BOXES=true to load grounded crop/phrase pairs." >&2
  exit 2
fi

export MODEL_NAME USE_BOXES
export CONFIG_PATH="${CONFIG_PATH:-configs/train_hycoslip_vit_s.py}"
export DATA_TARS="${DATA_TARS:-datasets/train/GRIT/processed/*.tar}"
export LOSS_TYPE="${LOSS_TYPE:-sigmoid}"
export MAIN_SCORE_TYPE="${MAIN_SCORE_TYPE:-root_angle}"
export EVAL_SCORE_TYPE="${EVAL_SCORE_TYPE:-main}"
export SIGMOID_LOGIT_BIAS_INIT="${SIGMOID_LOGIT_BIAS_INIT:--6.5}"

export ENTAILMENT_LOSS_TYPE="${ENTAILMENT_LOSS_TYPE:-radial_shadow}"
export ENTAILMENT_WEIGHT="${ENTAILMENT_WEIGHT:-0.15}"
export ANGULAR_TEMPERATURE_INIT="${ANGULAR_TEMPERATURE_INIT:-0.07}"
export ANGULAR_LOGIT_BIAS_INIT="${ANGULAR_LOGIT_BIAS_INIT:--6.5}"
export ANGULAR_REDUCTION="${ANGULAR_REDUCTION:-pair_mean}"
export ANGULAR_NEGATIVE_MINING="${ANGULAR_NEGATIVE_MINING:-all}"
export ANGULAR_TOPK="${ANGULAR_TOPK:-0}"
export RADIAL_SHADOW_SLOPE="${RADIAL_SHADOW_SLOPE:-8.0}"
export RADIAL_SHADOW_RADIAL_SCALE="${RADIAL_SHADOW_RADIAL_SCALE:-0.1}"
export RADIAL_SHADOW_RADIAL_WEIGHT="${RADIAL_SHADOW_RADIAL_WEIGHT:-1.0}"

export NORM_REG_TYPE="${NORM_REG_TYPE:-argent_space}"
export NORM_WEIGHT="${NORM_WEIGHT:-0.05}"

# Reproduce the default ViT-S/16 training budget unless explicitly overridden.
export NUM_GPUS="${NUM_GPUS:-4}"
export BATCH_SIZE="${BATCH_SIZE:-768}"
export NUM_ITERATIONS="${NUM_ITERATIONS:-500000}"
export LR="${LR:-5e-4}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.2}"
export WARMUP_STEPS="${WARMUP_STEPS:-4000}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export CHECKPOINT_PERIOD="${CHECKPOINT_PERIOD:-10000}"
export LOG_PERIOD="${LOG_PERIOD:-100}"
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-aurora_hycoslip_vit_s}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/../train_results}"

# Box fields are consumed by the mapper, so they must not be dropped.
export DROP_UNUSED_BOX_FIELDS="${DROP_UNUSED_BOX_FIELDS:-false}"

exec bash "${SCRIPT_DIR}/run_train_sigmeru.sh"
