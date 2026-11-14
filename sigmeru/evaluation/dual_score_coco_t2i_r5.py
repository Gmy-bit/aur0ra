"""COCO text-to-image Recall@5 only, preserving the existing retrieval pipeline.

The original zero-shot evaluator computes both retrieval directions and Flickr.
This adapter reuses its exact data encoding and recall implementations while
running only the requested COCO T->I R@5. No checkpoint or feature changes.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torchvision.transforms as T

from sigmeru.evaluation.catalog import DatasetCatalog
from sigmeru.evaluation.retrieval import (
    ZeroShotRetrievalEvaluator, _encode_dataset, _compute_recall,
)
from sigmeru.utils.evaluation import compute_similarity_scores


class DualScoreCocoT2IR5Evaluator(ZeroShotRetrievalEvaluator):
    def __init__(self, data_dir: str | Path, image_size: int = 224):
        super().__init__(datasets=["coco"], data_dir=data_dir, ks=[5], image_size=image_size)

    @torch.inference_mode()
    def __call__(self, model) -> dict[str, float]:
        model.eval()
        # Identical image transform to the original retrieval evaluator.
        image_transform = T.Compose([
            T.Resize((self._image_size, self._image_size), T.InterpolationMode.BICUBIC),
            T.ToTensor(),
        ])
        data_loader = DatasetCatalog.build("coco", self._data_dir, "test", image_transform)
        encoded = _encode_dataset(data_loader, model)
        image_feats = encoded["image_feats"].to(model.device)
        text_feats = encoded["text_feats"].to(model.device)
        image_ids = torch.tensor(encoded["image_ids"])
        text_ids = torch.tensor(encoded["text_ids"])

        text_to_image_retr = {}
        for ids, queries in zip(text_ids.split(256), text_feats.split(256)):
            scores = compute_similarity_scores(model, queries, image_feats)
            retrieval_order = scores.argsort(dim=1, descending=True).cpu()
            retrieved_image_ids = image_ids[retrieval_order]
            for query_id, ranked_image_ids in zip(ids, retrieved_image_ids):
                text_to_image_retr[query_id.item()] = ranked_image_ids.tolist()

        return {
            "coco_t2i_r5": _compute_recall(
                text_to_image_retr, encoded["text_to_image_gt"], 5
            )
        }
