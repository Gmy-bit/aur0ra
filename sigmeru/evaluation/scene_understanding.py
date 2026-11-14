from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms as T
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sigmeru.models import CLIPBaseline, HyCoCLIP, MERU, PHyCLIP
from sigmeru.tokenizer import Tokenizer
from sigmeru.utils.evaluation import compute_similarity_scores


_IMAGE_KEYS = [
    "image",
    "image_path",
    "img",
    "img_path",
    "path",
    "filename",
    "file_name",
]
_POSITIVE_KEYS = [
    "caption",
    "positive_caption",
    "pos_caption",
    "true_caption",
    "correct_caption",
    "sentence",
    "text",
]
_NEGATIVE_KEYS = [
    "negative_caption",
    "neg_caption",
    "false_caption",
    "hard_negative_caption",
    "foil_caption",
    "counterfactual_caption",
    "negative_sentence",
]


class PairwiseCaptionChoiceDataset(Dataset):
    """Image with a correct caption and a hard negative caption.

    Table 5 in HyCoCLIP evaluates scene understanding as a two-choice task: for
    each image, the model must assign a higher image-text score to the correct
    caption than to a caption where object/attribute information is perturbed.
    This dataset intentionally supports several common JSON/JSONL/CSV schemas so
    that VL-Checklist-Object and VG-Attribution files can be used without
    rewriting the evaluator.
    """

    def __init__(
        self,
        annotation_file: str | Path,
        image_root: str | Path | None = None,
        transform: T.Compose | None = None,
        max_samples: int | None = None,
        group_key: str | None = None,
    ):
        self.annotation_file = Path(annotation_file).expanduser().resolve()
        self.image_root = (
            Path(image_root).expanduser().resolve() if image_root else None
        )
        self.transform = transform

        if not self.annotation_file.exists():
            raise FileNotFoundError(
                f"Scene-understanding annotation file not found: {self.annotation_file}. "
                "Expected a JSON/JSONL/CSV file with image path, positive caption, "
                "and negative caption fields."
            )

        rows = _read_annotation_rows(self.annotation_file)
        samples = [_parse_pairwise_row(row, self.image_root, group_key) for row in rows]
        if max_samples is not None:
            samples = samples[: int(max_samples)]
        if not samples:
            raise RuntimeError(f"No valid samples found in {self.annotation_file}")
        self.samples = samples
        logger.info(
            f"Loaded {len(self.samples)} pairwise caption-choice samples from "
            f"{self.annotation_file}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        bbox = sample.get("bbox")
        if bbox is not None:
            x, y, w, h = bbox
            image = image.crop((x, y, x + w, y + h))
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "positive_caption": sample["positive_caption"],
            "negative_caption": sample["negative_caption"],
            "group": sample.get("group", "overall"),
        }


class SceneUnderstandingEvaluator:
    """Evaluate VL-Checklist-Object and VG-Attribution style benchmarks.

    Each benchmark is represented by a file of image/correct-caption/negative-
    caption triples. Accuracy is the percentage of samples where the correct
    caption receives a higher image-text similarity score than the hard negative.
    """

    def __init__(
        self,
        data_dir: str | Path,
        benchmarks: dict[str, dict[str, str]],
        image_size: int = 224,
        batch_size: int = 128,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        timeout: int = 30,
        max_samples: int | None = None,
        tie_is_correct: bool = False,
        group_key: str | None = None,
        include_overall: bool = True,
    ):
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.benchmarks = benchmarks
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers) and self.num_workers > 0
        self.timeout = int(timeout)
        self.max_samples = max_samples
        self.tie_is_correct = bool(tie_is_correct)
        self.group_key = group_key
        self.include_overall = bool(include_overall)
        self.image_transform = T.Compose(
            [
                T.Resize(image_size, T.InterpolationMode.BICUBIC),
                T.CenterCrop(image_size),
                T.ToTensor(),
            ]
        )

    @torch.inference_mode()
    def __call__(
        self, model: PHyCLIP | HyCoCLIP | MERU | CLIPBaseline
    ) -> dict[str, float]:
        model = model.eval()
        tokenizer = Tokenizer()
        results: dict[str, float] = {}

        for name, benchmark_cfg in self.benchmarks.items():
            annotation_file = _resolve_path(
                self.data_dir, benchmark_cfg["annotation_file"]
            )
            image_root = benchmark_cfg.get("image_root")
            resolved_image_root = (
                _resolve_path(self.data_dir, image_root) if image_root else None
            )
            dataset = PairwiseCaptionChoiceDataset(
                annotation_file=annotation_file,
                image_root=resolved_image_root,
                transform=self.image_transform,
                max_samples=self.max_samples,
                group_key=benchmark_cfg.get("group_key", self.group_key),
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

            num_correct = 0
            num_total = 0
            group_correct: dict[str, int] = defaultdict(int)
            group_total: dict[str, int] = defaultdict(int)
            for batch in tqdm(loader, desc=f"Evaluating {name}"):
                images = batch["image"].to(
                    model.device, non_blocking=torch.cuda.is_available()
                )
                positive_captions = list(batch["positive_caption"])
                negative_captions = list(batch["negative_caption"])

                if hasattr(model, "encode_image_for_eval"):
                    image_features = model.encode_image_for_eval(images)  # type: ignore[attr-defined]
                else:
                    image_features = model.encode_image(images, project=True)

                paired_captions = []
                for positive, negative in zip(positive_captions, negative_captions):
                    paired_captions.extend([positive, negative])
                text_features = model.encode_text(
                    tokenizer(paired_captions), project=True
                )  # type: ignore[arg-type]

                scores = compute_similarity_scores(model, image_features, text_features)
                batch_size = images.shape[0]
                row_indices = torch.arange(batch_size, device=scores.device)
                positive_scores = scores[row_indices, 2 * row_indices]
                negative_scores = scores[row_indices, 2 * row_indices + 1]
                if self.tie_is_correct:
                    correct = positive_scores >= negative_scores
                else:
                    correct = positive_scores > negative_scores

                num_correct += int(correct.sum().item())
                num_total += batch_size
                if "group" in batch:
                    groups = batch["group"]
                    correct_values = correct.detach().cpu().tolist()
                    for group, is_correct in zip(groups, correct_values):
                        group_name = str(group)
                        group_correct[group_name] += int(is_correct)
                        group_total[group_name] += 1

            accuracy = 100.0 * num_correct / num_total
            if self.include_overall or not group_total:
                results[name] = accuracy
                logger.info(
                    f"Scene understanding: {name}, {num_total} samples "
                    f"[acc.: {accuracy:.2f}%]"
                )
            for group_name in sorted(group_total):
                group_accuracy = (
                    100.0 * group_correct[group_name] / group_total[group_name]
                )
                result_name = f"{name}/{group_name}"
                results[result_name] = group_accuracy
                logger.info(
                    f"Scene understanding: {result_name}, "
                    f"{group_total[group_name]} samples [acc.: {group_accuracy:.2f}%]"
                )

        return results


def _read_annotation_rows(annotation_file: Path) -> list[dict[str, Any]]:
    suffix = annotation_file.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with annotation_file.open("r") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    if suffix == ".json":
        with annotation_file.open("r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        for key in ["annotations", "data", "samples", "examples", "items"]:
            if key in data and isinstance(data[key], list):
                return data[key]
        raise ValueError(
            f"Could not find a list of samples in {annotation_file}. "
            "Expected a JSON list or a dict with one of: annotations, data, samples, examples, items."
        )

    if suffix == ".csv":
        with annotation_file.open("r", newline="") as f:
            return list(csv.DictReader(f))

    raise ValueError(
        "Unsupported annotation file extension for "
        f"{annotation_file}; use .json, .jsonl, or .csv"
    )


def _parse_pairwise_row(
    row: dict[str, Any], image_root: Path | None, group_key: str | None = None
) -> dict[str, Any]:
    image_value = _get_first(row, _IMAGE_KEYS)
    positive_caption, negative_caption = _get_caption_pair(row)

    image_path = Path(str(image_value)).expanduser()
    if not image_path.is_absolute():
        if image_root is None:
            raise ValueError(
                f"Relative image path {image_path} requires image_root to be set in the benchmark config."
            )
        image_path = image_root / image_path
    image_path = image_path.resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Scene-understanding image not found: {image_path}")

    parsed = {
        "image_path": str(image_path),
        "positive_caption": str(positive_caption),
        "negative_caption": str(negative_caption),
        "bbox": _get_bbox(row),
    }
    if group_key is not None:
        parsed["group"] = str(_get_first(row, [group_key]))
    return parsed


def _get_first(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in row and row[key] not in [None, ""]:
            return row[key]
    raise ValueError(f"Could not find any of keys {keys} in row with keys {sorted(row.keys())}")


def _get_bbox(row: dict[str, Any]) -> tuple[float, float, float, float] | None:
    if all(key in row for key in ["bbox_x", "bbox_y", "bbox_w", "bbox_h"]):
        return (
            float(row["bbox_x"]),
            float(row["bbox_y"]),
            float(row["bbox_w"]),
            float(row["bbox_h"]),
        )

    bbox = row.get("bbox") or row.get("box")
    if isinstance(bbox, list) and len(bbox) == 4:
        return tuple(float(v) for v in bbox)  # type: ignore[return-value]

    return None


def _get_caption_pair(row: dict[str, Any]) -> tuple[str, str]:
    captions = row.get("captions") or row.get("sentences")
    label = row.get("label")
    if isinstance(captions, list) and len(captions) >= 2 and label is not None:
        label_idx = int(label)
        if label_idx not in [0, 1]:
            raise ValueError(f"Only binary caption choices are supported, got label={label!r}")
        return str(captions[label_idx]), str(captions[1 - label_idx])

    positive_caption = _get_first(row, _POSITIVE_KEYS)
    negative_caption = _get_negative_caption(row)
    return str(positive_caption), str(negative_caption)


def _get_negative_caption(row: dict[str, Any]) -> str:
    for key in _NEGATIVE_KEYS:
        if key in row and row[key] not in [None, ""]:
            return str(row[key])

    negatives = row.get("negative_captions") or row.get("negatives") or row.get("false_captions")
    if isinstance(negatives, list) and negatives:
        return str(negatives[0])

    raise ValueError(
        f"Could not find a negative caption in row with keys {sorted(row.keys())}. "
        f"Supported negative fields: {_NEGATIVE_KEYS}, negative_captions, negatives, false_captions."
    )


def _resolve_path(data_dir: Path, path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (data_dir / path).resolve()
