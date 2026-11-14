#!/usr/bin/env bash
set -euo pipefail

#############################################
# Zero-shot classification evaluation runner
#############################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

cd "${PROJECT_ROOT}"


#############################################
# Required configs
#############################################

CHECKPOINT_PATH=${CHECKPOINT_PATH:?"Set CHECKPOINT_PATH"}
TRAIN_CONFIG=${TRAIN_CONFIG:?"Set TRAIN_CONFIG"}


#############################################
# Evaluation configs
#############################################

EVAL_CONFIG=${EVAL_CONFIG:-configs/eval_zero_shot_classification.py}

EVAL_DATA_DIR=${EVAL_DATA_DIR:-"datasets/eval"}

EVAL_WORKERS=${EVAL_WORKERS:-4}

EVAL_TIMEOUT=${EVAL_TIMEOUT:-30}


#############################################
# Dataset list
#############################################

DATASETS=${DATASETS:-"imagenet,cifar10,cifar100,caltech101,stl10,food101,cub2011,cars,aircraft,pets,flowers102,dtd,eurosat,resisc45,sun397,country211"}


IFS=',' read -ra DATASET_ARRAY <<< "${DATASETS}"
echo "Datasets to evaluate:"
printf '%s\n' "${DATASET_ARRAY[@]}"

#############################################
# Batch size
#############################################

DEFAULT_BATCH_SIZE=${EVAL_BATCH_SIZE:-64}

SMALL_BATCH_SIZE=16

LARGE_DATASETS=(
    "imagenet"
    "sun397"
    "country211"
)


#############################################
# Logging
#############################################

LOG_DIR=${LOG_DIR:-"${PROJECT_ROOT}/eval_logs/$(basename "$(dirname "${CHECKPOINT_PATH}")")"}

mkdir -p "${LOG_DIR}"

SUMMARY_FILE="${LOG_DIR}/summary.txt"

echo "Zero-shot evaluation summary" > "${SUMMARY_FILE}"
echo "Checkpoint: ${CHECKPOINT_PATH}" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"


#############################################
# Check files
#############################################

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    echo "Checkpoint not found:"
    echo "${CHECKPOINT_PATH}"
    exit 1
fi


if [[ ! -f "${TRAIN_CONFIG}" ]]; then
    echo "Train config not found:"
    echo "${TRAIN_CONFIG}"
    exit 1
fi


if [[ ! -d "${EVAL_DATA_DIR}" ]]; then
    echo "Eval data directory not found:"
    echo "${EVAL_DATA_DIR}"
    exit 1
fi


#############################################
# Run evaluation
#############################################

FAILURES=()


for dataset in "${DATASET_ARRAY[@]}"; do

    dataset=$(echo "${dataset}" | xargs)

    if [[ -z "${dataset}" ]]; then
        continue
    fi


    LOG_FILE="${LOG_DIR}/${dataset}.log"


    echo "================================================"
    echo "Dataset: ${dataset}"
    echo "================================================"


    # choose batch size
    BATCH_SIZE=${DEFAULT_BATCH_SIZE}

    for large in "${LARGE_DATASETS[@]}"; do
        if [[ "${dataset}" == "${large}" ]]; then
            BATCH_SIZE=${SMALL_BATCH_SIZE}
        fi
    done


    # resume support
    if [[ -f "${LOG_FILE}" ]]; then
        if grep -q "All datasets finished successfully" "${LOG_FILE}"; then
            echo "Skip finished dataset: ${dataset}"
            continue
        fi
    fi



    set +e

    python scripts/evaluate.py \
        --config "${EVAL_CONFIG}" \
        --checkpoint-path "${CHECKPOINT_PATH}" \
        --train-config "${TRAIN_CONFIG}" \
        --datasets "${dataset}" \
        --eval-data-dir "${EVAL_DATA_DIR}" \
        --eval-batch-size "${BATCH_SIZE}" \
        --eval-workers "${EVAL_WORKERS}" \
        --eval-timeout "${EVAL_TIMEOUT}" \
        2>&1 | tee "${LOG_FILE}"


    status=${PIPESTATUS[0]}

    set -e



    if [[ ${status} -ne 0 ]]; then

        echo "FAILED: ${dataset}"

        FAILURES+=("${dataset}")

        echo "${dataset}: FAILED" >> "${SUMMARY_FILE}"

    else

        echo "DONE: ${dataset}"

        echo "${dataset}: OK" >> "${SUMMARY_FILE}"

    fi


done



#############################################
# Finish
#############################################

echo ""
echo "============================================"
echo "Evaluation finished"
echo "Logs: ${LOG_DIR}"
echo "============================================"


if [[ ${#FAILURES[@]} -gt 0 ]]; then

    echo "Failed datasets:"
    printf '%s\n' "${FAILURES[@]}"

    exit 1

fi


echo "All datasets completed successfully."