# AURORA 代码使用指南（中文）

本文档按照“环境安装 → GRIT 数据准备 → AURORA 训练 → 下游评测”的顺序说明当前代码快照的使用方法。训练与评测文档的组织参考了公开的 [HyCoCLIP](https://github.com/PalAvik/hycoclip) 和 [PHyCLIP](https://github.com/tksmatsubara/PHyCLIP) 仓库。

## AURORA 实现入口与训练配方

AURORA ViT-S/16 的配置文件为 `configs/train_hycoslip_vit_s.py`，正式启动脚本为 `scripts/run_train_hycoslip.sh`。代码中的历史 model-factory 名称 `hycoslip` 会路由到本仓库用于 AURORA 的 MERU-family 实现；它**不是**单独的 `hycoclip` 基线类。AURORA 启动脚本会自动设置 `MODEL_NAME=hycoslip` 和 `USE_BOXES=true`。

AURORA 训练配方包括：

- **根角对齐：** `MAIN_SCORE_TYPE=root_angle`，主图文对齐采用 Sigmoid 损失。
- **RAOC / HEC 监督：** 代码分支名称为 `radial_shadow`，实现径向—角度关系，并对匹配与未匹配的 batch 内蕴涵关系进行监督。
- **ARGENT-space 范数正则：** `NORM_REG_TYPE=argent_space`，`NORM_WEIGHT=0.05`。
- **视觉定位局部监督：** `USE_BOXES=true`，每条 GRIT 数据同时提供完整图像/文本对，以及采样得到的局部裁剪区域/短语对。

启动脚本默认使用论文中的 ViT-S/16 训练预算：4 张 GPU、global batch size 768、500,000 次迭代、AdamW（学习率 `5e-4`、betas `(0.9, 0.98)`、weight decay `0.2`）、4,000 步 warm-up 后余弦衰减，以及分辨率 224、scale `[0.5, 1.0]` 的随机缩放裁剪。

## 一、环境安装

`setup.py` 要求 Python >= 3.12。创建环境：

```bash
conda create -n aurora python=3.12 -y
conda activate aurora
```

先根据本机 CUDA 驱动安装匹配的 PyTorch 和 torchvision：

https://pytorch.org/get-started/locally/

再安装依赖和本仓库：

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

数据处理和分析脚本所需的可选依赖：

```bash
python -m pip install -r requirements-data.txt
```

检查环境：

```bash
python - <<'PY'
import torch, torchvision, timm, sigmeru
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("package:", sigmeru.__file__)
PY
```

## 二、GRIT 训练数据准备

训练需要处理后的 GRIT WebDataset TAR 分片。按照 GRIT 数据集发布方的说明获取原始 WebDataset，并放置在以下目录：

```text
datasets/
└── train/
    └── GRIT/
        ├── raw/
        │   ├── shard-000000.tar
        │   └── ...
        └── processed/
```

本仓库不分发 GRIT 原始数据。运行预处理脚本，提取训练 mapper 所需的局部裁剪区域和短语信息：

```bash
python prepare_GRIT_webdataset.py \
    --raw_webdataset_path datasets/train/GRIT/raw \
    --processed_webdataset_path datasets/train/GRIT/processed \
    --max_num_processes 12
```

检查处理结果：

```bash
find datasets/train/GRIT/processed -maxdepth 1 -name '*.tar' | head
```

AURORA 启动脚本默认读取 `datasets/train/GRIT/processed/*.tar`。数据存放在其他位置时，可通过 `DATA_TARS` 覆盖。

## 三、启动 AURORA 训练

### 3.1 ViT-S/16 正式训练命令

下面的命令与你提供的正式训练配置保持一致。`run_train_hycoslip.sh` 会自动补充 `MODEL_NAME=hycoslip`、`USE_BOXES=true`，以及 AURORA 的 RAOC/HEC 和范数正则默认值；其中范数正则明确设置为 `NORM_REG_TYPE=argent_space`、`NORM_WEIGHT=0.05`。

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

这里的 `BATCH_SIZE=768` 表示 DataLoader 每个优化步骤处理的全局 GRIT 原始记录数。由于 `USE_BOXES=true`，训练循环会将每条记录对应的完整图像/文本对与采样的局部裁剪区域/短语对拼接后送入模型。复现该配方时应保持相同的配置 batch size。

主要参数含义：

| 参数 | 作用 | 正式设置 |
|---|---|---:|
| `MAIN_SCORE_TYPE` | 图文主对齐评分 | `root_angle` |
| `EVAL_SCORE_TYPE` | 评测使用主评分 | `main` |
| `LOSS_TYPE` | 图文对齐损失 | `sigmoid` |
| `SIGMOID_LOGIT_BIAS_INIT` | 主 Sigmoid 损失偏置初始化 | `-6.5` |
| `ENTAILMENT_LOSS_TYPE` | RAOC/HEC 关系监督实现分支 | `radial_shadow` |
| `ENTAILMENT_WEIGHT` | 关系监督损失权重 | `0.15` |
| `ANGULAR_TEMPERATURE_INIT` | 角度项温度初始化 | `0.07` |
| `ANGULAR_LOGIT_BIAS_INIT` | 角度 Sigmoid 偏置初始化 | `-6.5` |
| `ANGULAR_REDUCTION` | 角度损失归约方式 | `pair_mean` |
| `ANGULAR_NEGATIVE_MINING` | 负样本选择方式 | `all` |
| `ANGULAR_TOPK` | top-k 负样本数量（`all` 时不启用） | `0` |
| `RADIAL_SHADOW_SLOPE` | 径向—角度边界斜率 | `8.0` |
| `RADIAL_SHADOW_RADIAL_SCALE` | 径向间隔归一化尺度 | `0.1` |
| `NORM_REG_TYPE` | 范数正则类型 | `argent_space` |
| `NORM_WEIGHT` | 范数正则权重 | `0.05` |

启动脚本还默认设置：

| 参数 | 默认值 |
|---|---:|
| `NUM_ITERATIONS` | 500000 |
| `LR` | 5e-4 |
| `WEIGHT_DECAY` | 0.2 |
| `WARMUP_STEPS` | 4000 |
| `NUM_WORKERS` | 8 |
| `CHECKPOINT_PERIOD` | 10000 |
| `LOG_PERIOD` | 100 |
| `MODEL_NAME` | `hycoslip` |
| `USE_BOXES` | `true` |
| `NORM_REG_TYPE` | `argent_space` |
| `NORM_WEIGHT` | 0.05 |

如需调整训练预算，可通过环境变量覆盖这些默认值。

**注意 `hycoslip` 与 `hycoclip` 的区别：** AURORA 正式训练必须使用 `MODEL_NAME=hycoslip`。不要将它替换为 `MODEL_NAME=hycoclip`；后者会选择独立的 HyCoCLIP 兼容类，不会使用 AURORA 的 radial-shadow 和范数正则分支。

### 3.2 其他基线和消融实验

通用启动脚本 `scripts/run_train_sigmeru.sh` 可以用于 root-angle baseline 或其他 model-factory 消融。例如，不使用 box、RAOC 和范数正则的 root-angle baseline：

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

其他配置包括：

```text
configs/train_clip_vit_{s,b,l}.py
configs/train_meru_vit_{s,b,l}.py
configs/train_hycoclip_vit_{s,b,l}.py
configs/train_phyclip_vit_{s,b,l}.py
configs/train_sigmeru_vit_{s,b,l}.py
```

PHyCLIP 的 direct-class 配置可以按照其配置驱动方式直接调用 `scripts/train.py`：

```bash
SIGMERU_OUTPUT_DIR=./train_results \
python scripts/train.py \
    --config configs/train_phyclip_vit_s.py \
    --num-gpus 4 \
    --checkpoint-period 10000 \
    --log-period 100
```

正式长跑前请核对所选配置和模型路由。尤其要区分：`hycoclip` 是 HyCoCLIP 基线，`hycoslip` 是本仓库 AURORA 使用的 model-factory 路由。

### 3.3 断点续训

从实验目录中的 `last_checkpoint.txt` 自动恢复：

```bash
RESUME=true \
CONFIG_PATH=configs/train_hycoslip_vit_s.py \
OUTPUT_ROOT=./train_results \
bash scripts/run_train_hycoslip.sh
```

从指定实验目录或 checkpoint 恢复：

```bash
RESUME_FROM=/path/to/run-directory \
TARGET_ITER=500000 \
CONFIG_PATH=configs/train_hycoslip_vit_s.py \
OUTPUT_ROOT=./train_results \
bash scripts/run_train_hycoslip.sh
```

直接调用 `scripts/train.py` 时，对应参数为 `--resume` 和 `--resume-from`。

## 四、模型评测

建议使用训练输出目录中保存的 `config.yaml` 作为 `--train-config`，以保证模型结构与训练时一致。

先设置路径：

```bash
RUN_DIR=/path/to/your/run-directory
CHECKPOINT_PATH="${RUN_DIR}/checkpoint_final.pth"
TRAIN_CONFIG="${RUN_DIR}/config.yaml"
EVAL_DATA_DIR=datasets/eval
```

### 4.1 Zero-shot classification

ImageNet 目录结构：

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

执行 ImageNet：

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

可以用逗号分隔多个分类数据集：

```bash
--datasets imagenet,food101,cifar10,cifar100
```

可用数据集名称以 `configs/eval_zero_shot_classification.py` 和 `sigmeru/evaluation/catalog.py` 为准。部分数据集通过 TFDS/Torchvision 下载；Stanford Cars 可能需要 Kaggle 凭据。`sun397`、`eurosat` 和 `resisc45` 在当前数据加载器中采用本地 ImageFolder 目录，需要按 `sigmeru/data/evaluation.py` 中的约定准备。

快捷脚本：

```bash
CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
TRAIN_CONFIG="${TRAIN_CONFIG}" \
EVAL_DATA_DIR="${EVAL_DATA_DIR}" \
DATASETS=imagenet \
bash scripts/run_eval_zero_shot_classification.sh
```

### 4.2 Zero-shot image-text retrieval

目录结构：

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

执行：

```bash
python scripts/evaluate.py \
    --config configs/eval_zero_shot_retrieval.py \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --train-config "${TRAIN_CONFIG}" \
    --eval-data-dir "${EVAL_DATA_DIR}"
```

或使用 shell：

```bash
CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
TRAIN_CONFIG="${TRAIN_CONFIG}" \
EVAL_DATA_DIR="${EVAL_DATA_DIR}" \
bash scripts/run_eval_zero_shot_retrieval_local.sh
```

### 4.3 Hierarchical classification

当前层级评测需要以下静态文件：

```text
assets/imagenet_synset/
├── all_synsets.pkl
├── all_ancestors_indices.pkl
└── imagenet_isa.txt
```

**这三个文件不在本次上传的压缩包中。** 需要先从匹配的上游资源获取或自行生成；尤其要确保 synset 顺序与评测器使用的 ImageNet 类别索引顺序一致。

安装 NLTK WordNet：

```bash
mkdir -p .nltk_data
python -m nltk.downloader -d .nltk_data wordnet omw-1.4
export NLTK_DATA="${PWD}/.nltk_data"
```

执行：

```bash
python scripts/evaluate.py \
    --config configs/eval_hierarchical_metrics.py \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --train-config "${TRAIN_CONFIG}" \
    --eval-data-dir "${EVAL_DATA_DIR}" \
    --datasets imagenet
```

### 4.4 SugarCrepe / Scene Understanding

确保 COCO `val2017` 图像已位于 `datasets/eval/coco/val2017`，然后准备 SugarCrepe：

```bash
python scripts/prepare_scene_understanding.py \
    --prepare sugarcrepe \
    --output-dir datasets/eval/scene_understanding \
    --sugarcrepe-image-root coco/val2017
```

运行：

```bash
python scripts/evaluate.py \
    --config configs/eval_sugarcrepe.py \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --train-config "${TRAIN_CONFIG}" \
    --eval-data-dir "${EVAL_DATA_DIR}"
```

也可以运行：

```bash
python scripts/run_sugarcrepe.py \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --train-config "${TRAIN_CONFIG}" \
    --eval-data-dir "${EVAL_DATA_DIR}"
```

如果需要联合评测 VL-Checklist-Object / Visual Genome，需要另外准备包含 `VG_100K/` 和 `VG_100K_2/` 的图像目录：

```bash
python scripts/prepare_scene_understanding.py \
    --prepare vlco \
    --output-dir datasets/eval/scene_understanding \
    --vlco-image-root /path/to/visual_genome_images
```

之后可以使用：

```bash
python scripts/evaluate.py \
    --config configs/eval_scene_understanding_table3.py \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --train-config "${TRAIN_CONFIG}" \
    --eval-data-dir "${EVAL_DATA_DIR}"
```

VG-Attribution 也提供了准备脚本：

```bash
python scripts/prepare_scene_understanding.py \
    --prepare vga \
    --output-dir datasets/eval/scene_understanding
```

该流程需要 `gdown`，并会从脚本中配置的公开来源下载数据。Table 5 的联合评测需要同时准备 VL-CO 和 VG-A，并使用 `configs/eval_scene_understanding_table5.py`。

### 4.5 可选的 VL-Checklist 外部包接口

`scripts/run_vl_checklist.py` 对接外部 `vl_checklist` 包。需要先按该项目自己的说明安装依赖并准备数据，再执行：

```bash
python scripts/run_vl_checklist.py \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --train-config "${TRAIN_CONFIG}" \
    --vl-checklist-config configs/vl_checklist_config.yaml
```

它与仓库内部 `SceneUnderstandingEvaluator` 的 pairwise caption-choice 评测是两套不同接口。

## 五、评测数据准备脚本

可以尝试用下面的脚本准备部分 TFDS/Torchvision 数据以及 COCO/Flickr30k：

```bash
EVAL_DATA_DIR=datasets/eval bash prepare_eval_datasets.sh
```

ImageNet 仍需手动放置或软链接。部分本地目录数据集和 benchmark 专用资产也需要单独准备。

## 六、输出文件

- 训练 checkpoint 和日志位于 `SIGMERU_OUTPUT_DIR` / `OUTPUT_ROOT` 指定的目录。
- `scripts/evaluate.py` 会将评测记录追加保存到 checkpoint 同目录下的 `evaluate.json`。
- `scripts/run_sugarcrepe.py` 会在 checkpoint 同目录下写入 `sugarcrepe_results/metrics.json`。
- Scene-understanding 数据准备结果位于 `datasets/eval/scene_understanding/`。

不要把数据集、checkpoint、训练日志、生成的 `config.yaml`、TensorBoard 文件或本地 NLTK 数据提交到匿名仓库。

## 七、参考仓库

- [HyCoCLIP](https://github.com/PalAvik/hycoclip)：GRIT 预处理、训练、零样本分类、检索和 WordNet 评测流程。
- [PHyCLIP](https://github.com/tksmatsubara/PHyCLIP)：依赖安装、配置驱动训练，以及 VL-Checklist / SugarCrepe 评测示例。

本次上传的压缩包不含根目录 `LICENSE` 文件，但部分源文件保留了上游版权声明。公开发布前应核实上游许可，并补齐适用的 License / Notice 文件。
