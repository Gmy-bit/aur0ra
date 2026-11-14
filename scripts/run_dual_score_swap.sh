#!/usr/bin/env bash
set -euo pipefail

# Run from any working directory. Uses only added files, never modifies
# checkpoints, training configs, original models, or original evaluators.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

HYCOCLIP_DIR="${HYCOCLIP_DIR:?Set HYCOCLIP_DIR to the HyCoCLIP experiment directory}"
AURORA_DIR="${AURORA_DIR:?Set AURORA_DIR to the AURORA experiment directory}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-datasets/eval}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/eval_logs/dual_score_swap}"
# Override e.g. MODELS="hycoclip", or SCORE_MODES="root_angle".
MODELS="${MODELS:-hycoclip aurora}"
SCORE_MODES="${SCORE_MODES:-lorentz_distance root_angle}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-30}"

for data_path in   "$EVAL_DATA_DIR/imagenet/val"   "$EVAL_DATA_DIR/coco/val2017"   "$EVAL_DATA_DIR/coco/annotations/captions_val2017.json"; do
  if [[ ! -e "$data_path" ]]; then
    echo "[ERROR] Missing evaluation data: $data_path" >&2
    exit 1
  fi
done

for model in $MODELS; do
  case "$model" in
    hycoclip) CKPT_DIR="$HYCOCLIP_DIR" ;;
    aurora) CKPT_DIR="$AURORA_DIR" ;;
    *) echo "[ERROR] Unsupported MODELS item: $model" >&2; exit 2 ;;
  esac
  CKPT="$CKPT_DIR/checkpoint_final.pth"
  CFG="$CKPT_DIR/config.yaml"
  for f in "$CKPT" "$CFG"; do
    if [[ ! -f "$f" ]]; then
      echo "[ERROR] Missing $model input: $f" >&2
      exit 1
    fi
  done
  LOG_DIR="$OUTPUT_ROOT/$model"
  mkdir -p "$LOG_DIR"
  RESULTS_JSON="$LOG_DIR/results.json"

  for score in $SCORE_MODES; do
    case "$score" in
      lorentz_distance|root_angle) ;;
      *) echo "[ERROR] Unsupported SCORE_MODES item: $score" >&2; exit 2 ;;
    esac

    echo "====== $model | ImageNet Top-1 | score=$score ======"
    python scripts/evaluate_dual_score_swap.py       --expected-model "$model"       --config configs/eval_zero_shot_classification.py       --checkpoint-path "$CKPT"       --train-config "$CFG"       --eval-score-type "$score"       --results-json-path "$RESULTS_JSON"       --datasets imagenet       --eval-data-dir "$EVAL_DATA_DIR"       --eval-batch-size "$EVAL_BATCH_SIZE"       --eval-workers "$EVAL_WORKERS"       --eval-timeout "$EVAL_TIMEOUT"       2>&1 | tee "$LOG_DIR/imagenet_${score}.log"

    echo "====== $model | COCO T->I R@5 | score=$score ======"
    python scripts/evaluate_dual_score_swap.py       --expected-model "$model"       --config configs/eval_dual_score_coco_t2i.py       --checkpoint-path "$CKPT"       --train-config "$CFG"       --eval-score-type "$score"       --results-json-path "$RESULTS_JSON"       --eval-data-dir "$EVAL_DATA_DIR"       2>&1 | tee "$LOG_DIR/coco_t2i_r5_${score}.log"
  done
  python scripts/summarize_dual_score_swap.py "$OUTPUT_ROOT" "$model"
done

echo "Results: $OUTPUT_ROOT"
