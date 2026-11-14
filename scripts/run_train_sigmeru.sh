#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"

OUTPUT_ROOT=${OUTPUT_ROOT:-"${PROJECT_ROOT}/train_results"}
export SIGMERU_OUTPUT_DIR="${SIGMERU_OUTPUT_DIR:-${OUTPUT_ROOT}}"

echo "[Info] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[Info] SIGMERU_OUTPUT_DIR=${SIGMERU_OUTPUT_DIR}"
echo "[Info] PYTHONPATH=${PYTHONPATH}"
echo "[Info] python=$(which python)"
python -c "import sigmeru; print('[Info] sigmeru =', sigmeru.__file__)"

CONFIG_PATH=${CONFIG_PATH:-configs/train_sigmeru_vit_s.py}
DATA_TARS=${DATA_TARS:-"datasets/train/GRIT/processed/*.tar"}
EXPERIMENT_TAG=${EXPERIMENT_TAG:-}

MODEL_NAME=${MODEL_NAME:-meru}          # clip | meru | hycoclip | phyclip
LOSS_TYPE=${LOSS_TYPE:-sigmoid}        # contrastive | sigmoid
USE_BOXES=${USE_BOXES:-false}

NUM_GPUS=${NUM_GPUS:-4}
NUM_MACHINES=${NUM_MACHINES:-1}
MACHINE_RANK=${MACHINE_RANK:-0}

BATCH_SIZE=${BATCH_SIZE:-768}
LR=${LR:-5e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.2}
WARMUP_STEPS=${WARMUP_STEPS:-4000}
NUM_ITERATIONS=${NUM_ITERATIONS:-500000}
NUM_WORKERS=${NUM_WORKERS:-8}
PRECISION_AMP=${PRECISION_AMP:-true}

# Hyperbolic geometry: learnable curvature clamped to [0.1, 10.0].
CURV_INIT=${CURV_INIT:-1.0}
LEARN_CURV=${LEARN_CURV:-true}
CURV_MIN=${CURV_MIN:-0.1}
CURV_MAX=${CURV_MAX:-10.0}
LORENTZ_EPS=${LORENTZ_EPS:-1e-8}
SUBSPACE_DIM=${SUBSPACE_DIM:-8}
PRODUCT_METRIC=${PRODUCT_METRIC:-l1}

# Main score for hyperbolic SigMERU/MERU.
MAIN_SCORE_TYPE=${MAIN_SCORE_TYPE:-root_angle}           # root_angle | lorentz_distance | rapidity
EVAL_SCORE_TYPE=${EVAL_SCORE_TYPE:-main}                 # main | root_angle | lorentz_distance | rapidity | angular_cos | typed_shadow | radial_shadow

# Main SigLIP-style sigmoid loss controls.
SIGMOID_NORMALIZATION=${SIGMOID_NORMALIZATION:-batch}    # batch | pairs
SIGMOID_NEGATIVES_PER_SAMPLE=${SIGMOID_NEGATIVES_PER_SAMPLE:-0}
# Manual bias reminders; these are not auto-selected by prior statistics.
# For B=768 pairwise sigmoid with bounded cosine-like scores, -log(B-1)≈-6.64,
# so -6.5 is a reasonable prior/default for root_angle and Euclidean cosine.
#   root_angle      : score=cos∠xOy, LOGIT_SCALE_INIT=10.0 -> try SIGMOID_LOGIT_BIAS_INIT=-6.5 first.
#   clip cosine     : score=cos,     LOGIT_SCALE_INIT=10.0 -> try SIGMOID_LOGIT_BIAS_INIT=-6.5 first.
#   lorentz_distance: score=-d_L,    LOGIT_SCALE_INIT=10.0 -> previous empirical setting SIGMOID_LOGIT_BIAS_INIT=10.0.
#   rapidity        : score=-rho,    LOGIT_SCALE_INIT=10.0 -> previous empirical setting SIGMOID_LOGIT_BIAS_INIT=10.0.
SIGMOID_LOGIT_BIAS_INIT=${SIGMOID_LOGIT_BIAS_INIT:--6.5}

# Angular entailment: always text(parent)->image(child).
ENTAILMENT_LOSS_TYPE=${ENTAILMENT_LOSS_TYPE:-none}       # none | positive_cos | contrastive_cos | sigmoid_cos | typed_shadow | radial_shadow | radial_shadow_decomposed
ENTAILMENT_WEIGHT=${ENTAILMENT_WEIGHT:-0.0}
# T2I exterior-angle contrastive/sigmoid temperature: 0.07 => scale=14.2857.
ANGULAR_TEMPERATURE_INIT=${ANGULAR_TEMPERATURE_INIT:-0.07}
ANGULAR_LOGIT_SCALE_INIT=${ANGULAR_LOGIT_SCALE_INIT:-10.0}  # fallback only if ANGULAR_TEMPERATURE_INIT is set empty/None in config
ANGULAR_LOGIT_BIAS_INIT=${ANGULAR_LOGIT_BIAS_INIT:--6.5}
ANGULAR_REDUCTION=${ANGULAR_REDUCTION:-pair_mean}            # pair_mean | balanced
ANGULAR_NEGATIVE_MINING=${ANGULAR_NEGATIVE_MINING:-all}      # all | topk
ANGULAR_TOPK=${ANGULAR_TOPK:-0}
# Exterior-angle reminders with tau=0.07:
#   positive_cos     uses only positive pairs: mean_i [1-cos alpha(t_i,v_i)].
#   contrastive_cos  is CLIP-like T2I CE((1/tau)*cos_alpha, targets), ignores ANGULAR_LOGIT_BIAS_INIT.
#   sigmoid_cos      is SigLIP-like pairwise sigmoid((1/tau)*cos_alpha + bias).
#                    For B=768, try ANGULAR_LOGIT_BIAS_INIT=-6.5 first;
#                    for a stricter zero-logit threshold near cos(alpha)=0.7, try -10.0.
#   typed_shadow     is the legacy sigmoid(scale*(psi_text-root_angle)+bias).
#   radial_shadow    is the R1 joint-min surrogate for the same typed order.
#   radial_shadow_decomposed separates pairwise angular membership from positive radial type order.
RADIAL_SHADOW_SLOPE=${RADIAL_SHADOW_SLOPE:-8.0}
RADIAL_SHADOW_RADIAL_SCALE=${RADIAL_SHADOW_RADIAL_SCALE:-0.1}
RADIAL_SHADOW_RADIAL_WEIGHT=${RADIAL_SHADOW_RADIAL_WEIGHT:-1.0}

# Directed anti-containment regularization. Ordinary text-shadow overlap is allowed;
# weight 0 completely skips its local text-text matrix.
SHADOW_OVERLAP_WEIGHT=${SHADOW_OVERLAP_WEIGHT:-0.0}
SHADOW_OVERLAP_SCALE=${SHADOW_OVERLAP_SCALE:-10.0}
SHADOW_OVERLAP_BIAS=${SHADOW_OVERLAP_BIAS:--1.0}

# Norm regularization.
NORM_REG_TYPE=${NORM_REG_TYPE:-none}                     # none | argent_space | argent_radius | inverse[_space] | inverse_square[_space] | softplus_neg[_space] | inverse_radius | inverse_square_radius | softplus_neg_radius
NORM_WEIGHT=${NORM_WEIGHT:-0.0}
NORM_REG_EPS=${NORM_REG_EPS:-1e-6}
NORM_REG_ALPHA=${NORM_REG_ALPHA:-5.0}

# Lightweight augmentation/regularization branches. Disabled by default.
TEXT_FEATURE_DROPOUT=${TEXT_FEATURE_DROPOUT:-0.0}
DISP_WEIGHT=${DISP_WEIGHT:-0.0}
DISP_SCALE=${DISP_SCALE:-10.0}
DISP_BIAS=${DISP_BIAS:--1.0}

CHECKPOINT_PERIOD=${CHECKPOINT_PERIOD:-10000}
LOG_PERIOD=${LOG_PERIOD:-100}
RUN_EVAL=${RUN_EVAL:-false}
RESUME=${RESUME:-false}
RESUME_FROM=${RESUME_FROM:-}
TARGET_ITER=${TARGET_ITER:-${NUM_ITERATIONS}}

DROP_UNUSED_BOX_FIELDS=${DROP_UNUSED_BOX_FIELDS:-true}
NON_BLOCKING_TO_DEVICE=${NON_BLOCKING_TO_DEVICE:-true}
ZERO_GRAD_SET_TO_NONE=${ZERO_GRAD_SET_TO_NONE:-true}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-4}
PERSISTENT_WORKERS=${PERSISTENT_WORKERS:-true}
GRADIENT_AS_BUCKET_VIEW=${GRADIENT_AS_BUCKET_VIEW:-true}

FUSED_ADAMW=${FUSED_ADAMW:-true}
COMPILE_MODEL=${COMPILE_MODEL:-false}
COMPILE_BACKEND=${COMPILE_BACKEND:-inductor}
COMPILE_MODE=${COMPILE_MODE:-default}
COMPILE_FULLGRAPH=${COMPILE_FULLGRAPH:-false}
COMPILE_DYNAMIC=${COMPILE_DYNAMIC:-false}
DDP_STATIC_GRAPH=${DDP_STATIC_GRAPH:-false}
DDP_FIND_UNUSED_PARAMETERS=${DDP_FIND_UNUSED_PARAMETERS:-false}
COMPILE_DDP_STATIC_GRAPH=${COMPILE_DDP_STATIC_GRAPH:-false}
COMPILE_FIND_UNUSED_PARAMETERS=${COMPILE_FIND_UNUSED_PARAMETERS:-true}

USE_TF32=${USE_TF32:-true}
FLOAT32_MATMUL_PRECISION=${FLOAT32_MATMUL_PRECISION:-high}
CUDA_MATMUL_TF32=${CUDA_MATMUL_TF32:-true}
CUDNN_TF32=${CUDNN_TF32:-true}
SYNC_CUDA_TIMING=${SYNC_CUDA_TIMING:-false}
LOG_ALL_RANKS_TO_STDERR=${LOG_ALL_RANKS_TO_STDERR:-false}
TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-OFF}
export TORCH_DISTRIBUTED_DEBUG

RESUME_ARGS=()
if [[ -n "${RESUME_FROM}" ]]; then
  RESUME_ARGS+=(--resume-from "${RESUME_FROM}")
elif [[ "${RESUME}" == "true" ]]; then
  RESUME_ARGS+=(--resume)
fi
if [[ -n "${TARGET_ITER}" ]]; then
  RESUME_ARGS+=(--target-iteration "${TARGET_ITER}")
fi

EXTRA_OVERRIDES=()
if [[ -n "${COMPILE_DYNAMIC}" ]]; then
  EXTRA_OVERRIDES+=(train.compile_dynamic="${COMPILE_DYNAMIC}")
fi

echo "[Info] CONFIG_PATH=${CONFIG_PATH}"
echo "[Info] DATA_TARS=${DATA_TARS}"
echo "[Info] EXPERIMENT_TAG=${EXPERIMENT_TAG}"
echo "[Info] MODEL_NAME=${MODEL_NAME} LOSS_TYPE=${LOSS_TYPE} USE_BOXES=${USE_BOXES}"
echo "[Info] MAIN_SCORE_TYPE=${MAIN_SCORE_TYPE} EVAL_SCORE_TYPE=${EVAL_SCORE_TYPE} SIGMOID_LOGIT_BIAS_INIT=${SIGMOID_LOGIT_BIAS_INIT}"
echo "[Info] ENTAILMENT_LOSS_TYPE=${ENTAILMENT_LOSS_TYPE} ENTAILMENT_WEIGHT=${ENTAILMENT_WEIGHT} ANG_TEMP_INIT=${ANGULAR_TEMPERATURE_INIT} ANG_SCALE_FALLBACK=${ANGULAR_LOGIT_SCALE_INIT} ANG_BIAS_INIT=${ANGULAR_LOGIT_BIAS_INIT}"
echo "[Info] ANGULAR_REDUCTION=${ANGULAR_REDUCTION} ANGULAR_NEGATIVE_MINING=${ANGULAR_NEGATIVE_MINING} ANGULAR_TOPK=${ANGULAR_TOPK}"
echo "[Info] NORM_REG_TYPE=${NORM_REG_TYPE} NORM_WEIGHT=${NORM_WEIGHT} NORM_REG_EPS=${NORM_REG_EPS} NORM_REG_ALPHA=${NORM_REG_ALPHA}"
echo "[Info] TEXT_FEATURE_DROPOUT=${TEXT_FEATURE_DROPOUT} DISP_WEIGHT=${DISP_WEIGHT} DISP_SCALE=${DISP_SCALE} DISP_BIAS=${DISP_BIAS}"
echo "[Info] RADIAL_SHADOW_SLOPE=${RADIAL_SHADOW_SLOPE} RADIAL_SHADOW_RADIAL_SCALE=${RADIAL_SHADOW_RADIAL_SCALE}"
echo "[Info] SHADOW_OVERLAP_WEIGHT=${SHADOW_OVERLAP_WEIGHT} SHADOW_OVERLAP_SCALE=${SHADOW_OVERLAP_SCALE} SHADOW_OVERLAP_BIAS=${SHADOW_OVERLAP_BIAS}"
echo "[Info] CURV_INIT=${CURV_INIT} LEARN_CURV=${LEARN_CURV} CURV_MIN=${CURV_MIN} CURV_MAX=${CURV_MAX}"
echo "[Info] RESUME=${RESUME} RESUME_FROM=${RESUME_FROM} TARGET_ITER=${TARGET_ITER}"

python scripts/train.py \
  --config "${CONFIG_PATH}" \
  --num-gpus "${NUM_GPUS}" \
  --num-machines "${NUM_MACHINES}" \
  --machine-rank "${MACHINE_RANK}" \
  --checkpoint-period "${CHECKPOINT_PERIOD}" \
  --log-period "${LOG_PERIOD}" \
  "${RESUME_ARGS[@]}" \
  dataset.tarfiles="['${DATA_TARS}']" \
  dataset.drop_unused_box_fields="${DROP_UNUSED_BOX_FIELDS}" \
  dataset.mapper.use_boxes="${USE_BOXES}" \
  model.model_name="${MODEL_NAME}" \
  model.loss_type="${LOSS_TYPE}" \
  model.use_boxes="${USE_BOXES}" \
  model.curv_init="${CURV_INIT}" \
  model.learn_curv="${LEARN_CURV}" \
  model.curv_min="${CURV_MIN}" \
  model.curv_max="${CURV_MAX}" \
  model.lorentz_eps="${LORENTZ_EPS}" \
  model.subspace_dim="${SUBSPACE_DIM}" \
  model.product_metric="${PRODUCT_METRIC}" \
  model.main_score_type="${MAIN_SCORE_TYPE}" \
  model.eval_score_type="${EVAL_SCORE_TYPE}" \
  model.sigmoid_logit_bias_init="${SIGMOID_LOGIT_BIAS_INIT}" \
  model.sigmoid_normalization="${SIGMOID_NORMALIZATION}" \
  model.sigmoid_negatives_per_sample="${SIGMOID_NEGATIVES_PER_SAMPLE}" \
  model.entailment_loss_type="${ENTAILMENT_LOSS_TYPE}" \
  model.entail_weight="${ENTAILMENT_WEIGHT}" \
  model.angular_temperature_init="${ANGULAR_TEMPERATURE_INIT}" \
  model.angular_logit_scale_init="${ANGULAR_LOGIT_SCALE_INIT}" \
  model.angular_logit_bias_init="${ANGULAR_LOGIT_BIAS_INIT}" \
  model.angular_reduction="${ANGULAR_REDUCTION}" \
  model.angular_negative_mining="${ANGULAR_NEGATIVE_MINING}" \
  model.angular_topk="${ANGULAR_TOPK}" \
  model.norm_reg_type="${NORM_REG_TYPE}" \
  model.norm_weight="${NORM_WEIGHT}" \
  model.norm_reg_eps="${NORM_REG_EPS}" \
  model.norm_reg_alpha="${NORM_REG_ALPHA}" \
  model.text_feature_dropout="${TEXT_FEATURE_DROPOUT}" \
  model.disp_weight="${DISP_WEIGHT}" \
  model.disp_scale="${DISP_SCALE}" \
  model.disp_bias="${DISP_BIAS}" \
  model.radial_shadow_slope="${RADIAL_SHADOW_SLOPE}" \
  model.radial_shadow_radial_scale="${RADIAL_SHADOW_RADIAL_SCALE}" \
  model.radial_shadow_radial_weight="${RADIAL_SHADOW_RADIAL_WEIGHT}" \
  model.shadow_overlap_weight="${SHADOW_OVERLAP_WEIGHT}" \
  model.shadow_overlap_scale="${SHADOW_OVERLAP_SCALE}" \
  model.shadow_overlap_bias="${SHADOW_OVERLAP_BIAS}" \
  optim.optimizer.lr="${LR}" \
  optim.optimizer.weight_decay="${WEIGHT_DECAY}" \
  optim.optimizer.fused="${FUSED_ADAMW}" \
  optim.lr_scheduler.warmup_steps="${WARMUP_STEPS}" \
  train.total_batch_size="${BATCH_SIZE}" \
  train.num_iterations="${TARGET_ITER}" \
  train.num_workers="${NUM_WORKERS}" \
  train.amp="${PRECISION_AMP}" \
  train.run_eval="${RUN_EVAL}" \
  train.experiment_tag="${EXPERIMENT_TAG}" \
  train.non_blocking_to_device="${NON_BLOCKING_TO_DEVICE}" \
  train.zero_grad_set_to_none="${ZERO_GRAD_SET_TO_NONE}" \
  train.prefetch_factor="${PREFETCH_FACTOR}" \
  train.persistent_workers="${PERSISTENT_WORKERS}" \
  train.ddp.static_graph="${DDP_STATIC_GRAPH}" \
  train.ddp.find_unused_parameters="${DDP_FIND_UNUSED_PARAMETERS}" \
  train.ddp.gradient_as_bucket_view="${GRADIENT_AS_BUCKET_VIEW}" \
  train.compile_model="${COMPILE_MODEL}" \
  train.compile_backend="${COMPILE_BACKEND}" \
  train.compile_mode="${COMPILE_MODE}" \
  train.compile_fullgraph="${COMPILE_FULLGRAPH}" \
  train.compile_ddp_static_graph="${COMPILE_DDP_STATIC_GRAPH}" \
  train.compile_find_unused_parameters="${COMPILE_FIND_UNUSED_PARAMETERS}" \
  train.use_tf32="${USE_TF32}" \
  train.float32_matmul_precision="${FLOAT32_MATMUL_PRECISION}" \
  train.cuda_matmul_tf32="${CUDA_MATMUL_TF32}" \
  train.cudnn_tf32="${CUDNN_TF32}" \
  train.sync_cuda_timing="${SYNC_CUDA_TIMING}" \
  train.log_all_ranks_to_stderr="${LOG_ALL_RANKS_TO_STDERR}" \
  "${EXTRA_OVERRIDES[@]}"
