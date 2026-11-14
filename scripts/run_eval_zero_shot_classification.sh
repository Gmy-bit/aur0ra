#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"

# ============================================================
# Main knobs
# ============================================================
# Example:
#   CHECKPOINT_PATH=/path/to/checkpoint_final.pth \
#   TRAIN_CONFIG=/path/to/config.yaml \
#   DATASETS=imagenet \
#   bash scripts/run_eval_zero_shot_classification.sh

CHECKPOINT_PATH=${CHECKPOINT_PATH:?"Please set CHECKPOINT_PATH=/path/to/checkpoint.pth"}
TRAIN_CONFIG=${TRAIN_CONFIG:?"Please set TRAIN_CONFIG=/path/to/train config.yaml or train_*.py"}

# Use this ImageNet-only config by default. You can also point to
# configs/eval_zero_shot_classification.py and control the subset with DATASETS.
EVAL_CONFIG=${EVAL_CONFIG:-configs/eval_zero_shot_classification.py}

# Parent directory containing imagenet/, coco/, flickr30k/, etc.
EVAL_DATA_DIR=${EVAL_DATA_DIR:-datasets/eval}

# Comma-separated subset available in the eval config, e.g.
#   imagenet
#   imagenet,cifar10
DATASETS=${DATASETS:-imagenet}

EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-512}
EVAL_WORKERS=${EVAL_WORKERS:-8}
EVAL_TIMEOUT=${EVAL_TIMEOUT:-30}

# Optional: choose one GPU for evaluation.
# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python scripts/evaluate.py \
  --config "${EVAL_CONFIG}" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --train-config "${TRAIN_CONFIG}" \
  --datasets "${DATASETS}" \
  --eval-data-dir "${EVAL_DATA_DIR}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --eval-workers "${EVAL_WORKERS}" \
  --eval-timeout "${EVAL_TIMEOUT}"
