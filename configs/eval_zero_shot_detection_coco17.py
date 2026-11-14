from sigmeru.config import LazyCall as L
from sigmeru.evaluation.detection import (
    COCO_17_NOVEL_CLASSES,
    ZeroShotCoco17DetectionEvaluator,
)


evaluator = L(ZeroShotCoco17DetectionEvaluator)(
    data_dir="datasets/eval",
    split="val",
    class_names=COCO_17_NOVEL_CLASSES,
    prompts=[
        "a photo of a {}.",
        "a photo of the {}.",
        "a cropped photo of a {}.",
        "a close-up photo of a {}.",
    ],
    image_size=224,
    batch_size=256,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    timeout=30,
    min_box_size=1.0,
    include_crowd=False,
    max_regions=None,
)
