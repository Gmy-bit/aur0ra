# AURORA: Hyperbolic Vision–Language Representation Learning

This repository provides the training and evaluation code used for AURORA, together with compatible CLIP-, MERU/SigMERU-, HyCoCLIP-, and PHyCLIP-style baselines. The usage flow follows the practical organization of the public [HyCoCLIP](https://github.com/PalAvik/hycoclip) and [PHyCLIP](https://github.com/tksmatsubara/PHyCLIP) repositories: environment setup, GRIT preparation, training, and task-specific evaluation.

## AURORA implementation and default recipe

The AURORA ViT-S/16 recipe is provided as `configs/train_hycoslip_vit_s.py` and launched with `scripts/run_train_hycoslip.sh`. The historical model-factory key `hycoslip` selects the MERU-family implementation used by this AURORA code path; it is **not** the separate `hycoclip` baseline class. The AURORA launcher sets this key and enables grounded box/crop inputs automatically.

The recipe combines:

- **Root-angle alignment:** `MAIN_SCORE_TYPE=root_angle` with sigmoid image–text alignment.
- **RAOC / HEC supervision:** the implementation branch is named `radial_shadow`; it uses the radial–angular relation with matched and unmatched in-batch entailment supervision.
- **ARGENT-space norm regularization:** `NORM_REG_TYPE=argent_space` and `NORM_WEIGHT=0.05`.
- **Grounded local supervision:** `USE_BOXES=true`, so each GRIT record supplies the full image/text pair and its sampled grounded crop/phrase pair.

The AURORA launcher defaults to the paper's ViT-S/16 recipe: 4 GPUs, global batch size 768, 500,000 iterations, AdamW with learning rate `5e-4`, betas `(0.9, 0.98)`, weight decay `0.2`, 4,000 warm-up steps, cosine decay, and 224-pixel random resized crops with scale `[0.5, 1.0]`.

## Repository layout

```text
.
├── configs/                     # Training and evaluation configurations
├── scripts/                     # Training, evaluation, and analysis entry points
├── sigmeru/                     # Models, encoders, datasets, losses, and evaluators
├── assets/                      # Optional static evaluation assets (some supplied separately)
├── prepare_GRIT_webdataset.py   # GRIT preprocessing
├── prepare_eval_datasets.sh     # Optional evaluation-data preparation helper
├── README.md
├── USAGE_CN.md
├── requirements.txt
└── requirements-data.txt
```

## 1. Environment setup

The package metadata requires **Python 3.12 or newer**. Create an environment:

```bash
conda create -n aurora python=3.12 -y
conda activate aurora
```

Install a PyTorch and torchvision build compatible with your CUDA runtime and GPU driver. Use the official selector:

- https://pytorch.org/get-started/locally/

Then install the Python dependencies and this package:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Optional dependencies for data preparation and analysis:

```bash
python -m pip install -r requirements-data.txt
```

Check the environment:

```bash
python - <<'PY'
import torch
import torchvision
import timm
import sigmeru

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("package:", sigmeru.__file__)
PY
```

## 2. Training

### 2.1 Prepare GRIT

Training expects processed GRIT WebDataset shards. Download the raw GRIT WebDataset files according to the dataset provider's instructions and place the `.tar` shards under:

```text
datasets/
└── train/
    └── GRIT/
        ├── raw/
        │   ├── shard-000000.tar
        │   └── ...
        └── processed/
```

This repository does not redistribute GRIT. Preprocess the raw shards to extract the grounded crop/phrase information used by the training mapper:

```bash
python prepare_GRIT_webdataset.py \
    --raw_webdataset_path datasets/train/GRIT/raw \
    --processed_webdataset_path datasets/train/GRIT/processed \
    --max_num_processes 12
```

Confirm that the processed directory contains TAR shards:

```bash
find datasets/train/GRIT/processed -maxdepth 1 -name '*.tar' | head
```

The AURORA launcher reads `datasets/train/GRIT/processed/*.tar` by default. Override this with `DATA_TARS` if the data is stored elsewhere.

### 2.2 Train AURORA ViT-S/16

The following command follows the AURORA training recipe. `run_train_hycoslip.sh` supplies `MODEL_NAME=hycoslip`, `USE_BOXES=true`, and the AURORA loss/regularization defaults, including `NORM_REG_TYPE=argent_space` and `NORM_WEIGHT=0.05`.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
CONFIG_PATH=configs/train_hycoslip_vit_s.py \
DATA_TARS='datasets/train/GRIT/processed/*.tar' \
NUM_GPUS=4 \
BATCH_SIZE=768 \
MAIN_SCORE_TYPE=root_angle \
EVAL_SCORE_TYPE=main \
LOSS_TYPE=sigmoid \
SIGMOID_LOGIT_BIAS_INIT=-6.5 \
ENTAILMENT_LOSS_TYPE=radial_shadow \
ENTAILMENT_WEIGHT=0.15 \
ANGULAR_TEMPERATURE_INIT=0.07 \
ANGULAR_LOGIT_BIAS_INIT=-6.5 \
ANGULAR_REDUCTION=pair_mean \
ANGULAR_NEGATIVE_MINING=all \
ANGULAR_TOPK=0 \
RADIAL_SHADOW_SLOPE=8.0 \
RADIAL_SHADOW_RADIAL_SCALE=0.1 \
NORM_REG_TYPE=argent_space \
NORM_WEIGHT=0.05 \
EXPERIMENT_TAG=aurora_vit_s \
OUTPUT_ROOT=./train_results \
bash scripts/run_train_hycoslip.sh
```

The launcher also defaults to `NUM_ITERATIONS=500000`, `LR=5e-4`, `WEIGHT_DECAY=0.2`, `WARMUP_STEPS=4000`, `NUM_WORKERS=8`, `CHECKPOINT_PERIOD=10000`, and `LOG_PERIOD=100`. Override any of these through environment variables when needed.

In this implementation, `BATCH_SIZE=768` is the global number of GRIT records loaded by the data loader. With `USE_BOXES=true`, each record contributes both its full image/text pair and its sampled grounded crop/phrase pair to the model forward pass. Keep the same configured batch size when reproducing this recipe.

**Important model-name distinction:** use `MODEL_NAME=hycoslip` for this AURORA recipe. Do not substitute `MODEL_NAME=hycoclip`: the latter selects the separate HyCoCLIP compatibility class, which does not use the AURORA radial-shadow and norm-regularization branches.

### 2.3 Other model configurations and ablations

The generic launcher `scripts/run_train_sigmeru.sh` exposes the full set of model-factory overrides. It can be used for root-angle baselines or other ablations. For example, a no-box root-angle baseline can be launched with:

```bash
CONFIG_PATH=configs/train_sigmeru_vit_s.py \
DATA_TARS='datasets/train/GRIT/processed/*.tar' \
MODEL_NAME=meru \
LOSS_TYPE=sigmoid \
USE_BOXES=false \
MAIN_SCORE_TYPE=root_angle \
ENTAILMENT_LOSS_TYPE=none \
ENTAILMENT_WEIGHT=0.0 \
NORM_REG_TYPE=none \
NORM_WEIGHT=0.0 \
NUM_GPUS=4 \
BATCH_SIZE=768 \
NUM_ITERATIONS=500000 \
OUTPUT_ROOT=./train_results \
bash scripts/run_train_sigmeru.sh
```

Other configurations include:

```text
configs/train_clip_vit_{s,b,l}.py
configs/train_meru_vit_{s,b,l}.py
configs/train_hycoclip_vit_{s,b,l}.py
configs/train_phyclip_vit_{s,b,l}.py
configs/train_sigmeru_vit_{s,b,l}.py
```

For PHyCLIP's direct-class config, launch `scripts/train.py` as in the upstream configuration-driven workflow:

```bash
SIGMERU_OUTPUT_DIR=./train_results \
python scripts/train.py \
    --config configs/train_phyclip_vit_s.py \
    --num-gpus 4 \
    --checkpoint-period 10000 \
    --log-period 100
```

Check the selected config and model route before starting a long run. In particular, `hycoclip` is the HyCoCLIP baseline, while `hycoslip` is the AURORA model-factory route used by the provided launcher.

### 2.4 Checkpoints and resuming

The output root is controlled by `OUTPUT_ROOT` (mapped to `SIGMERU_OUTPUT_DIR`). Each run creates an experiment directory containing files such as:

```text
<run-directory>/
├── config.yaml
├── log-rank0.txt
├── checkpoint_XXXXXXXX.pth
├── checkpoint_final.pth
└── last_checkpoint.txt
```

Resume the AURORA run from its latest checkpoint:

```bash
RESUME=true \
CONFIG_PATH=configs/train_hycoslip_vit_s.py \
OUTPUT_ROOT=./train_results \
bash scripts/run_train_hycoslip.sh
```

Or resume from an explicit experiment directory/checkpoint:

```bash
RESUME_FROM=/path/to/run-directory \
TARGET_ITER=500000 \
CONFIG_PATH=configs/train_hycoslip_vit_s.py \
OUTPUT_ROOT=./train_results \
bash scripts/run_train_hycoslip.sh
```

For direct use of `scripts/train.py`, the corresponding options are `--resume` and `--resume-from`.

## 3. Evaluation

Use the training `config.yaml` saved inside the experiment directory as `--train-config`. This preserves the model architecture and configuration used to create the checkpoint.

Set these paths once:

```bash
RUN_DIR=/path/to/your/run-directory
CHECKPOINT_PATH="${RUN_DIR}/checkpoint_final.pth"
TRAIN_CONFIG="${RUN_DIR}/config.yaml"
EVAL_DATA_DIR=datasets/eval
```

### 3.1 Zero-shot image classification

ImageNet should be arranged in Torchvision `ImageFolder` format:

```text
datasets/eval/
└── imagenet/
    └── val/
        ├── n01440764/
        │   └── *.JPEG
        ├── n01443537/
        │   └── *.JPEG
        └── ...
```

Run ImageNet evaluation:

```bash
python scripts/evaluate.py \
    --config configs/eval_zero_shot_classification.py \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --train-config "${TRAIN_CONFIG}" \
    --eval-data-dir "${EVAL_DATA_DIR}" \
    --datasets imagenet \
    --eval-batch-size 64 \
    --eval-workers 8
```

To evaluate several supported classification datasets, pass a comma-separated list, for example:

```bash
--datasets imagenet,food101,cifar10,cifar100
```

The classification config contains the supported dataset names and prompt templates. Some datasets are downloaded/cached through TensorFlow Datasets or Torchvision; others require manual preparation. Stanford Cars may require Kaggle credentials. The local-folder datasets `sun397`, `eurosat`, and `resisc45` must be arranged in the folder layout expected by `sigmeru/data/evaluation.py`.

A convenience launcher is also provided:

```bash
CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
TRAIN_CONFIG="${TRAIN_CONFIG}" \
EVAL_DATA_DIR="${EVAL_DATA_DIR}" \
DATASETS=imagenet \
bash scripts/run_eval_zero_shot_classification.sh
```

### 3.2 Zero-shot image–text retrieval

The retrieval evaluator expects COCO and Flickr30k under the evaluation root:

```text
datasets/eval/
├── coco/
│   ├── val2017/
│   └── annotations/
│       └── captions_val2017.json
└── flickr30k/
    ├── dataset_flickr30k.json
    └── flickr30k_images/
```

Run:

```bash
python scripts/evaluate.py \
    --config configs/eval_zero_shot_retrieval.py \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --train-config "${TRAIN_CONFIG}" \
    --eval-data-dir "${EVAL_DATA_DIR}"
```

Or use the shell wrapper:

```bash
CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
TRAIN_CONFIG="${TRAIN_CONFIG}" \
EVAL_DATA_DIR="${EVAL_DATA_DIR}" \
bash scripts/run_eval_zero_shot_retrieval_local.sh
```

### 3.3 Hierarchical classification

The WordNet-based ImageNet evaluator expects the ImageNet validation set and the following static files:

```text
assets/imagenet_synset/
├── all_synsets.pkl
├── all_ancestors_indices.pkl
└── imagenet_isa.txt
```

**These three files are not included in the supplied archive.** Obtain or generate the matching assets before running this evaluator; the synset ordering must correspond to the ImageNet class-index ordering used by the evaluator.

Install/download the NLTK WordNet corpus:

```bash
mkdir -p .nltk_data
python -m nltk.downloader -d .nltk_data wordnet omw-1.4
export NLTK_DATA="${PWD}/.nltk_data"
```

Then run:

```bash
python scripts/evaluate.py \
    --config configs/eval_hierarchical_metrics.py \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --train-config "${TRAIN_CONFIG}" \
    --eval-data-dir "${EVAL_DATA_DIR}" \
    --datasets imagenet
```

The shell wrapper checks for the required assets and NLTK corpus:

```bash
CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
TRAIN_CONFIG="${TRAIN_CONFIG}" \
EVAL_DATA_DIR="${EVAL_DATA_DIR}" \
bash scripts/run_eval_hierarchical_classification_local.sh
```

### 3.4 SugarCrepe and scene-understanding evaluation

Prepare SugarCrepe annotations. COCO `val2017` images should already exist under `datasets/eval/coco/val2017`:

```bash
python scripts/prepare_scene_understanding.py \
    --prepare sugarcrepe \
    --output-dir datasets/eval/scene_understanding \
    --sugarcrepe-image-root coco/val2017
```

Run the dedicated evaluator:

```bash
python scripts/evaluate.py \
    --config configs/eval_sugarcrepe.py \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --train-config "${TRAIN_CONFIG}" \
    --eval-data-dir "${EVAL_DATA_DIR}"
```

Alternatively, use the wrapper:

```bash
python scripts/run_sugarcrepe.py \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --train-config "${TRAIN_CONFIG}" \
    --eval-data-dir "${EVAL_DATA_DIR}"
```

For the combined scene-understanding configuration, prepare the required VL-Checklist-Object / Visual Genome images as well. The image root must contain `VG_100K/` and `VG_100K_2/`:

```bash
python scripts/prepare_scene_understanding.py \
    --prepare vlco \
    --output-dir datasets/eval/scene_understanding \
    --vlco-image-root /path/to/visual_genome_images
```

Then run the corresponding config:

```bash
python scripts/evaluate.py \
    --config configs/eval_scene_understanding_table3.py \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --train-config "${TRAIN_CONFIG}" \
    --eval-data-dir "${EVAL_DATA_DIR}"
```

The VG-Attribution preparation path is also available:

```bash
python scripts/prepare_scene_understanding.py \
    --prepare vga \
    --output-dir datasets/eval/scene_understanding
```

It requires `gdown` and downloads the source files from the locations configured in the preparation script. For the combined VL-CO + VG-A evaluation, prepare both benchmarks and use `configs/eval_scene_understanding_table5.py`.

### 3.5 Optional VL-Checklist package evaluation

`scripts/run_vl_checklist.py` is an integration with the external `vl_checklist` package. Install and prepare that package and its data according to its own instructions, then run:

```bash
python scripts/run_vl_checklist.py \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --train-config "${TRAIN_CONFIG}" \
    --vl-checklist-config configs/vl_checklist_config.yaml
```

This optional integration is separate from the in-repository `SceneUnderstandingEvaluator` used by the scene-understanding configs above.

## 4. Optional evaluation-data helper

The helper below attempts to prepare several TFDS/Torchvision datasets and COCO/Flickr30k files:

```bash
EVAL_DATA_DIR=datasets/eval bash prepare_eval_datasets.sh
```

ImageNet still needs to be placed or symlinked manually. Some local-folder datasets and benchmark-specific assets also require manual preparation; see the dataset layout and evaluator source code.

## 5. Output files

- Training checkpoints and logs are written below `SIGMERU_OUTPUT_DIR` / `OUTPUT_ROOT`.
- `scripts/evaluate.py` appends evaluation records to `evaluate.json` beside the checkpoint.
- `scripts/run_sugarcrepe.py` writes `sugarcrepe_results/metrics.json` beside the checkpoint.
- Scene-understanding preparation files are written below `datasets/eval/scene_understanding`.

Do not commit datasets, checkpoints, run logs, generated configs, TensorBoard files, or local NLTK data to the anonymous source repository.


## 6. References and implementation lineage

The training/evaluation workflow is organized in the style of these public repositories:

- [HyCoCLIP](https://github.com/PalAvik/hycoclip): GRIT preprocessing, training, zero-shot classification, retrieval, and WordNet evaluation.
- [PHyCLIP](https://github.com/tksmatsubara/PHyCLIP): dependency installation, configuration-driven training, and additional VL-Checklist / SugarCrepe evaluation examples.

The supplied archive does not contain a root `LICENSE` file, although several source files retain upstream copyright headers. Verify the applicable upstream licenses and include the appropriate license/notice files before public release.
