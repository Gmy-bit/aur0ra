# ---------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------

# Modified from github.com/facebookresearch/meru

from __future__ import annotations

from pathlib import Path

import torch
import torchvision.transforms as T
from loguru import logger
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

from sigmeru.evaluation.catalog import DatasetCatalog
from sigmeru.evaluation.class_names import CLASS_NAMES
from sigmeru.models import MERU, CLIPBaseline, PHyCLIP, HyCoCLIP
from sigmeru.tokenizer import Tokenizer
from sigmeru.utils.evaluation import compute_similarity_scores, process_text_features


class ZeroShotClassificationEvaluator:
    """
    Evaluate trained models for zero-shot image classification, wherein the entire
    model is transferred to the downstream task without additional training. This
    protocol is similar to CLIP: the classifier weights are constructed by encoding
    text prompts of class labels using text encoder.

    Reference: CLIP paper (https://arxiv.org/abs/2103.00020)
    """

    def __init__(
        self,
        datasets_and_prompts: dict[str, list[str]],
        data_dir: str | Path,
        image_size: int = 224,
        batch_size: int = 512,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        timeout: int = 30,
    ):
        """
        Args:
            datasets_and_prompts: Dictionary mapping between dataset name and
                a list of prompt templates to fill using its class names. Add
                a single `{}` in prompt to fill with class name. Datasets
                should be among supported datasets in `DatasetCatalog`.
            data_dir: Path to directory containing sub-directories of all datasets
                that are supported by the dataset catalog. For ImageNet arranged as
                data_dir/imagenet/val/<wnid>/*.JPEG, pass data_dir.
            image_size: Resize and crop images to this size for evaluation. We
                resize the smaller image edge (keeping aspect ratio same) using
                bicubic interpolation, and take a square center crop.
            batch_size: Batch size used for image feature extraction.
            num_workers: Number of DataLoader workers.
            pin_memory: Whether DataLoader should pin CPU memory.
            persistent_workers: Keep DataLoader workers alive after iterator creation.
            timeout: DataLoader worker timeout in seconds. Set 0 to disable.
        """
        self._datasets_and_prompts = datasets_and_prompts
        self._data_dir = Path(data_dir).resolve()
        self._batch_size = int(batch_size)
        self._num_workers = int(num_workers)
        self._pin_memory = bool(pin_memory)
        self._persistent_workers = bool(persistent_workers) and self._num_workers > 0
        self._timeout = int(timeout)
        self._image_transform = T.Compose(
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

        # Collect results per task in this dict.
        results_dict = {}

        for dname, prompts in self._datasets_and_prompts.items():
            logger.info(f"Starting evaluation for dataset: {dname}")
            logger.info(f"Evaluation data root: {self._data_dir}")

            # ----------------------------------------------------------------
            # Make zero-shot classifier using class name and prompts.
            # ----------------------------------------------------------------
            class_names = CLASS_NAMES[dname]

            # Collect text features of each class.
            all_class_feats: list[torch.Tensor] = []

            for name in tqdm(class_names, desc=f"Building {dname} classifier"):
                class_prompts = [_pt.format(name) for _pt in prompts]
                class_prompt_tokens = tokenizer(class_prompts)

                # Keep the original PHyCLIP/MERU zero-shot protocol:
                # 1) encode all prompt templates before manifold projection;
                # 2) ensemble those prompt features in process_text_features;
                # 3) obtain exactly one classifier vector per class.
                #
                # Do not call encode_text_for_eval/build_text_classifier_features here:
                # those helpers are useful elsewhere, but using them in this loop can
                # bypass the original prompt-ensemble path and leave an unwanted
                # [num_prompts, dim] axis in the classifier.
                prompt_feats = model.encode_text(class_prompt_tokens, project=False)  # type: ignore[arg-type]
                class_feats = process_text_features(model, prompt_feats)

                if class_feats.dim() != 1:
                    raise RuntimeError(
                        "Zero-shot classifier construction must return one vector per class, "
                        f"got class {name!r} feature shape {tuple(class_feats.shape)}."
                    )
                all_class_feats.append(class_feats)

            # shape: (num_classes, embed_dim)
            classifier = torch.stack(all_class_feats, dim=0)
            logger.info(f"Classifier shape: {classifier.size()}")

            # Extract image features and labels from the test split of required dataset.
            dataset = DatasetCatalog.build(
                dname, self._data_dir, "test", self._image_transform
            )
            logger.info(f"Built dataset {dname} with {len(dataset)} images")

            loader = DataLoader(
                dataset,
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                pin_memory=self._pin_memory,
                persistent_workers=self._persistent_workers,
                drop_last=False,
                timeout=self._timeout,
            )

            logger.info(f"Starting feature extraction for dataset: {dname}")
            image_feats, labels = _encode_dataset(loader, model, project=True)

            # Features returned by this function will be on CPU, move to device.
            image_feats = image_feats.to(model.device, non_blocking=torch.cuda.is_available())

            # Measure model performance according to accuracy metric of the dataset.
            acc_meter = MulticlassAccuracy(DatasetCatalog.NUM_CLASSES[dname])

            # Evaluate in small batches of 256 instances.
            for _feats, _labels in zip(image_feats.split(256), labels.split(256)):
                # Compute pairwise similarity depending on model type.
                scores = compute_similarity_scores(model, _feats, classifier)

                acc_meter(scores.cpu(), _labels)

            accuracy = acc_meter.compute() * 100.0
            results_dict[dname] = accuracy

            logger.info(
                f"Zero-shot evaluation: {dname}, {len(image_feats)} images, "
                f"{len(class_names)} classes [acc.: {accuracy:.1f}%] "
            )

            # Clear variables to free memory before next dataset.
            del image_feats, labels, classifier, all_class_feats, dataset, loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results_dict


def _encode_dataset(
    data_loader: DataLoader,
    model: PHyCLIP | HyCoCLIP | MERU | CLIPBaseline,
    project: bool,
):
    """
    Extract image features and labels for a given dataset using the given model.

    Args:
        data_loader: PyTorch dataset or dataloader that serves instances/batches
            of `(image, label)` tuples.
        model: Model that implements `encode_image` method to extract features.
        project: Input argument to `model.encode_image`.
    """

    # Collect batches of extracted image features and labels (as-is from loader).
    all_image_feats, all_labels = [], []

    try:
        for batch_idx, (images, labels) in enumerate(
            tqdm(data_loader, desc="Extracting image feats")
        ):
            with torch.inference_mode():
                try:
                    images = images.to(
                        model.device,
                        non_blocking=torch.cuda.is_available(),
                    )
                    if hasattr(model, "encode_image_for_eval"):
                        image_feats = model.encode_image_for_eval(images)  # type: ignore[attr-defined]
                    else:
                        image_feats = model.encode_image(images, project)
                    all_image_feats.append(image_feats.cpu())
                    all_labels.append(labels.cpu())
                except Exception as e:
                    logger.exception(f"Error processing batch {batch_idx}")
                    raise

    except Exception as e:
        logger.error(f"Error during dataset encoding: {e}")
        if len(all_image_feats) == 0:
            raise RuntimeError("No batches were successfully processed") from e

    logger.info(f"Extracted {len(all_image_feats)} batches of image features.")
    return torch.cat(all_image_feats, dim=0), torch.cat(all_labels, dim=0)
