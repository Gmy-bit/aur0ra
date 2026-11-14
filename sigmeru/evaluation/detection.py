from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import torch
import torchvision.transforms as T
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sigmeru.models import CLIPBaseline, HyCoCLIP, MERU, PHyCLIP
from sigmeru.tokenizer import Tokenizer
from sigmeru.utils.evaluation import compute_similarity_scores, process_text_features


COCO_17_NOVEL_CLASSES: list[str] = [
    "airplane",
    "bus",
    "cat",
    "dog",
    "cow",
    "elephant",
    "umbrella",
    "tie",
    "snowboard",
    "skateboard",
    "cup",
    "knife",
    "cake",
    "couch",
    "keyboard",
    "sink",
    "scissors",
]


class CocoGroundTruthBoxRegions(Dataset):
    """COCO GT-box region crops for closed-set zero-shot object detection.

    The HyCoCLIP paper evaluates zero-shot object detection by using ground-truth
    COCO boxes as region proposals and predicting the category of each proposed
    region. This dataset exposes those proposals as cropped image regions.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        class_names: list[str],
        transform: Callable | None = None,
        min_box_size: float = 1.0,
        include_crowd: bool = False,
        max_regions: int | None = None,
    ):
        self.root = Path(root)
        self.split = split
        self.class_names = class_names
        self.transform = transform
        self.min_box_size = float(min_box_size)
        self.include_crowd = bool(include_crowd)

        annotations_path = self.root / "annotations" / f"instances_{split}2017.json"
        if not annotations_path.exists():
            raise FileNotFoundError(f"COCO instances annotation file not found: {annotations_path}")

        with annotations_path.open("r") as f:
            coco = json.load(f)

        category_name_to_id = {cat["name"]: int(cat["id"]) for cat in coco["categories"]}
        missing = [name for name in class_names if name not in category_name_to_id]
        if missing:
            raise ValueError(f"Class names are missing from COCO categories: {missing}")

        self.category_id_to_label = {
            category_name_to_id[name]: label for label, name in enumerate(class_names)
        }
        self.images = {int(img["id"]): img for img in coco["images"]}

        samples = []
        for ann in coco["annotations"]:
            category_id = int(ann["category_id"])
            if category_id not in self.category_id_to_label:
                continue
            if not self.include_crowd and int(ann.get("iscrowd", 0)):
                continue

            x, y, w, h = [float(v) for v in ann["bbox"]]
            if w < self.min_box_size or h < self.min_box_size:
                continue

            image_info = self.images[int(ann["image_id"])]
            image_path = self.root / f"{split}2017" / image_info["file_name"]
            if not image_path.exists():
                raise FileNotFoundError(f"COCO image not found: {image_path}")

            samples.append(
                {
                    "image_id": int(ann["image_id"]),
                    "annotation_id": int(ann["id"]),
                    "image_path": image_path,
                    "bbox": (x, y, w, h),
                    "label": self.category_id_to_label[category_id],
                }
            )

        if max_regions is not None:
            samples = samples[: int(max_regions)]

        if not samples:
            raise RuntimeError(
                f"No COCO GT-box regions found for split={split!r} and classes={class_names}."
            )

        self.samples = samples
        logger.info(
            f"Built COCO GT-box region dataset from {annotations_path}: "
            f"{len(self.samples)} regions, {len(self.class_names)} classes."
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        x, y, w, h = sample["bbox"]
        left = max(0.0, x)
        top = max(0.0, y)
        right = min(float(image.width), x + w)
        bottom = min(float(image.height), y + h)
        region = image.crop((left, top, right, bottom))

        if self.transform is not None:
            region = self.transform(region)

        return region, torch.tensor(sample["label"], dtype=torch.long)


class ZeroShotCoco17DetectionEvaluator:
    """Zero-shot object detection on COCO 17 novel categories with GT boxes.

    This follows the protocol described in HyCoCLIP Table 3: COCO ground-truth
    bounding boxes are treated as object proposals, each cropped region is scored
    against text classifiers for the 17 novel categories, and per-class AP is
    computed over all proposed regions.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "val",
        class_names: list[str] | None = None,
        prompts: list[str] | None = None,
        image_size: int = 224,
        batch_size: int = 256,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        timeout: int = 30,
        min_box_size: float = 1.0,
        include_crowd: bool = False,
        max_regions: int | None = None,
    ):
        self.data_dir = Path(data_dir).resolve()
        self.split = split
        self.class_names = class_names or COCO_17_NOVEL_CLASSES
        self.prompts = prompts or [
            "a photo of a {}.",
            "a photo of the {}.",
            "a cropped photo of a {}.",
            "a close-up photo of a {}.",
        ]
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers) and self.num_workers > 0
        self.timeout = int(timeout)
        self.min_box_size = float(min_box_size)
        self.include_crowd = bool(include_crowd)
        self.max_regions = max_regions
        self.image_transform = T.Compose(
            [
                T.Resize(image_size, T.InterpolationMode.BICUBIC),
                T.CenterCrop(image_size),
                T.ToTensor(),
            ]
        )

    @torch.inference_mode()
    def __call__(self, model: PHyCLIP | HyCoCLIP | MERU | CLIPBaseline) -> dict[str, float]:
        model = model.eval()
        classifier = self._build_text_classifier(model)
        dataset = CocoGroundTruthBoxRegions(
            root=self.data_dir / "coco",
            split=self.split,
            class_names=self.class_names,
            transform=self.image_transform,
            min_box_size=self.min_box_size,
            include_crowd=self.include_crowd,
            max_regions=self.max_regions,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
            timeout=self.timeout,
        )

        all_scores: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        for images, labels in tqdm(loader, desc="Scoring COCO GT-box regions"):
            images = images.to(model.device, non_blocking=torch.cuda.is_available())
            if hasattr(model, "encode_image_for_eval"):
                image_features = model.encode_image_for_eval(images)  # type: ignore[attr-defined]
            else:
                image_features = model.encode_image(images, project=True)
            scores = compute_similarity_scores(model, image_features, classifier)
            all_scores.append(scores.cpu())
            all_labels.append(labels.cpu())

        scores = torch.cat(all_scores, dim=0)
        labels = torch.cat(all_labels, dim=0)
        aps = _compute_closed_set_region_ap(scores, labels, self.class_names)
        mean_ap = sum(aps.values()) / len(aps)

        results = {"AP": mean_ap * 100.0}
        for name in self.class_names:
            results[f"AP-{name}"] = aps[name] * 100.0

        logger.info(
            f"COCO 17 novel GT-box zero-shot detection: {len(labels)} regions, "
            f"AP={results['AP']:.2f}"
        )
        return results

    def _build_text_classifier(
        self, model: PHyCLIP | HyCoCLIP | MERU | CLIPBaseline
    ) -> torch.Tensor:
        tokenizer = Tokenizer()
        class_features = []
        for name in tqdm(self.class_names, desc="Building COCO-17 classifier"):
            prompt_texts = [prompt.format(name) for prompt in self.prompts]
            tokens = tokenizer(prompt_texts)
            prompt_features = model.encode_text(tokens, project=False)  # type: ignore[arg-type]
            class_feature = process_text_features(model, prompt_features)
            if class_feature.dim() != 1:
                raise RuntimeError(
                    f"Expected one classifier vector for class {name!r}, got {tuple(class_feature.shape)}."
                )
            class_features.append(class_feature)

        classifier = torch.stack(class_features, dim=0)
        logger.info(f"COCO-17 text classifier shape: {tuple(classifier.shape)}")
        return classifier


def _compute_closed_set_region_ap(
    scores: torch.Tensor, labels: torch.Tensor, class_names: list[str]
) -> dict[str, float]:
    num_classes = len(class_names)
    if scores.ndim != 2 or scores.shape[1] != num_classes:
        raise ValueError(
            f"Expected scores with shape [num_regions, {num_classes}], got {tuple(scores.shape)}."
        )
    if labels.ndim != 1 or labels.shape[0] != scores.shape[0]:
        raise ValueError(
            f"Expected labels with shape [{scores.shape[0]}], got {tuple(labels.shape)}."
        )

    ap_by_class = {}
    for class_idx, class_name in enumerate(class_names):
        class_scores = scores[:, class_idx]
        positives = labels == class_idx
        num_positives = int(positives.sum().item())
        if num_positives == 0:
            ap_by_class[class_name] = 0.0
            continue

        order = torch.argsort(class_scores, descending=True)
        sorted_positives = positives[order].float()
        true_positives = torch.cumsum(sorted_positives, dim=0)
        ranks = torch.arange(1, len(sorted_positives) + 1, dtype=torch.float32)
        precision = true_positives / ranks
        ap = (precision * sorted_positives).sum() / num_positives
        ap_by_class[class_name] = float(ap.item())

    return ap_by_class
