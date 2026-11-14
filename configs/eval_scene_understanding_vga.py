from sigmeru.config import LazyCall as L
from sigmeru.evaluation.scene_understanding import SceneUnderstandingEvaluator


evaluator = L(SceneUnderstandingEvaluator)(
    data_dir="datasets/eval",
    benchmarks={
        "VG-A": {
            "annotation_file": "scene_understanding/vg_attribution.jsonl",
            "image_root": "scene_understanding/images",
        },
    },
    image_size=224,
    batch_size=128,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    timeout=30,
    max_samples=None,
    tie_is_correct=False,
)
