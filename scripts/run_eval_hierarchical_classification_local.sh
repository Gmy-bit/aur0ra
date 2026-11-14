#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export NLTK_DATA=${NLTK_DATA:-"${PROJECT_ROOT}/.nltk_data"}

cd "${PROJECT_ROOT}"

CHECKPOINT_PATH=${CHECKPOINT_PATH:?"Set CHECKPOINT_PATH"}
TRAIN_CONFIG=${TRAIN_CONFIG:?"Set TRAIN_CONFIG"}
EVAL_CONFIG=${EVAL_CONFIG:-configs/eval_hierarchical_metrics.py}
EVAL_DATA_DIR=${EVAL_DATA_DIR:-"datasets/eval"}
LOG_DIR=${LOG_DIR:-"${PROJECT_ROOT}/eval_logs/$(basename "$(dirname "${CHECKPOINT_PATH}")")"}
LOG_FILE=${LOG_FILE:-"${LOG_DIR}/hierarchical_classification.log"}

required_paths=(
    "${CHECKPOINT_PATH}"
    "${TRAIN_CONFIG}"
    "${EVAL_CONFIG}"
    "${EVAL_DATA_DIR}/imagenet/val"
    "${PROJECT_ROOT}/assets/imagenet_synset/all_synsets.pkl"
    "${PROJECT_ROOT}/assets/imagenet_synset/all_ancestors_indices.pkl"
    "${PROJECT_ROOT}/assets/imagenet_synset/imagenet_isa.txt"
)

for path in "${required_paths[@]}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Required path not found: ${path}" >&2
        exit 1
    fi
done

if ! python -c "from nltk.corpus import wordnet as wn; next(wn.all_synsets())" >/dev/null 2>&1; then
    echo "NLTK WordNet is unavailable under NLTK_DATA=${NLTK_DATA}" >&2
    exit 1
fi

mkdir -p "${LOG_DIR}"

echo "ImageNet hierarchical classification evaluation"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Data: ${EVAL_DATA_DIR}"
echo "NLTK data: ${NLTK_DATA}"
echo "Log: ${LOG_FILE}"

set +e
python scripts/evaluate.py \
    --config "${EVAL_CONFIG}" \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --train-config "${TRAIN_CONFIG}" \
    --datasets imagenet \
    --eval-data-dir "${EVAL_DATA_DIR}" \
    2>&1 | tee "${LOG_FILE}"
status=${PIPESTATUS[0]}
set -e

if [[ ${status} -ne 0 ]]; then
    echo "Hierarchical classification evaluation failed. See ${LOG_FILE}" >&2
    exit "${status}"
fi

echo "Hierarchical classification evaluation completed successfully."
