#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}" && pwd)"
EVAL_ROOT="${EVAL_DATA_DIR:-${PROJECT_ROOT}/datasets/eval}"
export EVAL_DATA_DIR="${EVAL_ROOT}"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

mkdir -p "$EVAL_ROOT"

echo "============================================================"
echo "[1/7] Install/check download helpers"
echo "============================================================"
python -c "import tensorflow_datasets" || { echo "Install tensorflow-datasets first: pip install tensorflow-datasets" >&2; exit 1; }

echo "============================================================"
echo "[2/7] Download TFDS-backed classification datasets"
echo "============================================================"
python - <<'PY'
import os
import tensorflow_datasets as tfds

root = os.environ["EVAL_DATA_DIR"]

tfds_names = [
    "food101",
    "cifar10",
    "cifar100",
    "dtd",
    "oxford_iiit_pet",
    "caltech101",
    "oxford_flowers102",
    "stl10",
    "mnist",
    "clevr",
    "patch_camelyon",
]

for name in tfds_names:
    print(f"\n[TFDS] preparing {name} -> {root}")
    try:
        builder = tfds.builder(name, data_dir=root)
        builder.download_and_prepare()
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {repr(e)}")
PY

echo "[MANUAL] sun397, eurosat, and resisc45 use local ImageFolder layouts in this codebase."
echo "         Prepare them as sun397_raw/, eurosat_raw/, and resisc45_raw/test/ under $EVAL_ROOT."

echo "============================================================"
echo "[3/7] Trigger Torchvision/custom dataset downloads"
echo "============================================================"
python - <<'PY'
import os
from sigmeru.evaluation.catalog import DatasetCatalog

root = os.environ["EVAL_DATA_DIR"]

# cars 单独用 Kaggle CLI 处理；ImageNet 不自动下载；COCO/Flickr30k 手动处理。
datasets = [
    "cub2011",
    "aircraft",
    "country211",
    "sst2",
]

for dname in datasets:
    print(f"\n[DatasetCatalog] preparing {dname}")
    try:
        dset = DatasetCatalog.build(dname, root, "test", transform=None)
        print(f"[OK] {dname}: len={len(dset)}")
    except Exception as e:
        print(f"[FAIL] {dname}: {repr(e)}")
PY

echo "============================================================"
echo "[4/7] Download COCO val2017 + captions annotation"
echo "============================================================"
mkdir -p "$EVAL_ROOT/coco"
cd "$EVAL_ROOT/coco"

if [ ! -d "val2017" ]; then
  wget -c http://images.cocodataset.org/zips/val2017.zip
  unzip -n val2017.zip
else
  echo "[SKIP] COCO val2017 already exists"
fi

if [ ! -f "annotations/captions_val2017.json" ]; then
  wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
  unzip -n annotations_trainval2017.zip
else
  echo "[SKIP] COCO captions_val2017.json already exists"
fi

echo "============================================================"
echo "[5/7] Prepare Flickr30k Karpathy JSON + images"
echo "============================================================"
mkdir -p "$EVAL_ROOT/flickr30k"
cd "$EVAL_ROOT/flickr30k"

if [ ! -f "dataset_flickr30k.json" ]; then
  wget -c https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip -O caption_datasets.zip
  unzip -j -n caption_datasets.zip dataset_flickr30k.json
else
  echo "[SKIP] dataset_flickr30k.json already exists"
fi

# 尝试用 Kaggle 镜像下载 Flickr30k 图片；如失败，后面会给出手动提示。
if [ ! -d "flickr30k_images" ]; then
  if command -v kaggle >/dev/null 2>&1; then
    echo "[INFO] trying Kaggle Flickr30k mirror..."
    kaggle datasets download -d adityajn105/flickr30k -p "$EVAL_ROOT/flickr30k" --unzip || true

    # 尝试自动找到包含 jpg 的目录，并软链接为代码需要的 flickr30k_images
    IMG_PARENT="$(find "$EVAL_ROOT/flickr30k" -type f \( -name '*.jpg' -o -name '*.JPG' \) | head -n 1 | xargs -r dirname || true)"
    if [ -n "${IMG_PARENT:-}" ] && [ -d "$IMG_PARENT" ]; then
      ln -sfn "$IMG_PARENT" "$EVAL_ROOT/flickr30k/flickr30k_images"
      echo "[OK] linked Flickr30k images: $EVAL_ROOT/flickr30k/flickr30k_images -> $IMG_PARENT"
    else
      echo "[WARN] could not locate Flickr30k jpg images after Kaggle download"
    fi
  else
    echo "[WARN] kaggle CLI not found; skip Flickr30k image download"
  fi
else
  echo "[SKIP] flickr30k_images already exists"
fi

echo "============================================================"
echo "[6/7] Prepare Stanford Cars via Kaggle"
echo "============================================================"
mkdir -p "$EVAL_ROOT/cars"

if command -v kaggle >/dev/null 2>&1; then
  kaggle datasets download \
    -d emanuelriquelmem/stanford-cars-pytorch \
    -p "$EVAL_ROOT/cars" \
    --unzip || true
else
  echo "[WARN] kaggle CLI not found; skip Stanford Cars"
fi

echo "[INFO] validating Stanford Cars with DatasetCatalog..."
cd "$PROJECT_ROOT"
python - <<'PY'
import os
from sigmeru.evaluation.catalog import DatasetCatalog

root = os.environ["EVAL_DATA_DIR"]
try:
    dset = DatasetCatalog.build("cars", root, "test", transform=None)
    print(f"[OK] cars: len={len(dset)}")
except Exception as e:
    print("[WARN] cars validation failed:")
    print(repr(e))
    print("""
Expected final structure is usually like:

${EVAL_ROOT}/cars/stanford_cars/
├── cars_test_annos_withlabels.mat
├── cars_train/
├── cars_test/
└── devkit/
    ├── cars_meta.mat
    ├── cars_test_annos.mat
    ├── cars_train_annos.mat
    └── ...
""")
PY

echo "============================================================"
echo "[7/7] Check manual datasets: ImageNet and Flickr30k"
echo "============================================================"

if [ -d "$EVAL_ROOT/imagenet/val" ]; then
  echo "[OK] ImageNet val folder exists: $EVAL_ROOT/imagenet/val"
else
  echo "[NEED MANUAL] ImageNet is not auto-downloaded."
  echo "Put or symlink ImageNet ImageFolder here:"
  echo "  $EVAL_ROOT/imagenet/val/<synset>/*.JPEG"
  echo ""
  echo "Example:"
  echo "  ln -s /path/to/your/imagenet $EVAL_ROOT/imagenet"
fi

if [ -f "$EVAL_ROOT/flickr30k/dataset_flickr30k.json" ] && [ -d "$EVAL_ROOT/flickr30k/flickr30k_images" ]; then
  echo "[OK] Flickr30k looks ready"
else
  echo "[NEED MANUAL] Flickr30k requires:"
  echo "  $EVAL_ROOT/flickr30k/dataset_flickr30k.json"
  echo "  $EVAL_ROOT/flickr30k/flickr30k_images/*.jpg"
fi

echo "============================================================"
echo "Done. Eval root: $EVAL_ROOT"
echo "============================================================"
