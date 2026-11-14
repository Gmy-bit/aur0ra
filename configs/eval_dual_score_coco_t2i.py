"""COCO T->I Recall@5 only. Follows the original extraction/ranking path."""
from sigmeru.config import LazyCall as L
from sigmeru.evaluation.dual_score_coco_t2i_r5 import DualScoreCocoT2IR5Evaluator

evaluator = L(DualScoreCocoT2IR5Evaluator)(
    data_dir="eval",
    image_size=224,
)
