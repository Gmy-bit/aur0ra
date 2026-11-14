#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

cd "${PROJECT_ROOT}"

CHECKPOINT_PATH=${CHECKPOINT_PATH:?"Set CHECKPOINT_PATH"}
TRAIN_CONFIG=${TRAIN_CONFIG:?"Set TRAIN_CONFIG"}
EVAL_CONFIG=${EVAL_CONFIG:-configs/eval_zero_shot_retrieval.py}
EVAL_DATA_DIR=${EVAL_DATA_DIR:-"datasets/eval"}
LOG_DIR=${LOG_DIR:-"${PROJECT_ROOT}/eval_logs/$(basename "$(dirname "${CHECKPOINT_PATH}")")"}
LOG_FILE=${LOG_FILE:-"${LOG_DIR}/zero_shot_retrieval.log"}

required_paths=(
    "${EVAL_DATA_DIR}/coco/val2017"
    "${EVAL_DATA_DIR}/coco/annotations/captions_val2017.json"
    "${EVAL_DATA_DIR}/flickr30k/dataset_flickr30k.json"
    "${EVAL_DATA_DIR}/flickr30k/flickr30k_images"
)

for path in "${CHECKPOINT_PATH}" "${TRAIN_CONFIG}" "${EVAL_CONFIG}" "${required_paths[@]}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Required path not found: ${path}" >&2
        exit 1
    fi
done

mkdir -p "${LOG_DIR}"

echo "Zero-shot image-text retrieval evaluation"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Data: ${EVAL_DATA_DIR}"
echo "Log: ${LOG_FILE}"

set +e
python scripts/evaluate.py \
    --config "${EVAL_CONFIG}" \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --train-config "${TRAIN_CONFIG}" \
    --eval-data-dir "${EVAL_DATA_DIR}" \
    2>&1 | tee "${LOG_FILE}"
status=${PIPESTATUS[0]}
set -e

if [[ ${status} -ne 0 ]]; then
    echo "Zero-shot retrieval evaluation failed. See ${LOG_FILE}" >&2
    exit "${status}"
fi

echo "Zero-shot retrieval evaluation completed successfully."
