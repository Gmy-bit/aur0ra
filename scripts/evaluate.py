# ---------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------

"""
Evaluate a trained model using implementations from `sigmeru.evaluation` module.

This version keeps the original PHyCLIP/MERU evaluation protocol but adds
lightweight command-line controls for local evaluation:
  * --datasets: run a comma-separated subset such as "imagenet".
  * --eval-data-dir: override evaluator.data_dir without editing config files.
  * --eval-batch-size / --eval-workers / --eval-timeout: override DataLoader knobs.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Iterable

import torch
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from sigmeru.config import LazyConfig, LazyFactory
from sigmeru.utils.checkpointing import CheckpointManager

parser = argparse.ArgumentParser(description=__doc__)
_AA = parser.add_argument
_AA(
    "--config",
    help=(
        "Path to an evaluation config file (.py). If not provided, all evaluations "
        "(classification, retrieval, hierarchical) will be run."
    ),
)
_AA(
    "--checkpoint-path",
    required=True,
    help="Path to checkpoint of a trained HyCoCLIP/MERU/CLIP/SigMERU model.",
)
_AA(
    "--train-config",
    required=True,
    help="Path to train config (.yaml/.yml/.py) for given checkpoint.",
)
_AA(
    "--datasets",
    default="",
    help=(
        "Comma-separated dataset subset for evaluators with datasets_and_prompts, "
        "for example: 'imagenet' or 'imagenet,cifar10'. Empty means use config as-is."
    ),
)
_AA(
    "--eval-data-dir",
    default="",
    help=(
        "Override evaluator.data_dir. For ImageNet with val class folders at "
        "/path/evaldata/imagenet/val, pass /path/evaldata."
    ),
)
_AA(
    "--eval-batch-size",
    type=int,
    default=None,
    help="Override evaluator.batch_size if the evaluator supports it.",
)
_AA(
    "--eval-workers",
    type=int,
    default=None,
    help="Override evaluator.num_workers if the evaluator supports it.",
)
_AA(
    "--eval-timeout",
    type=int,
    default=None,
    help="Override evaluator.timeout if the evaluator supports it. Use 0 to disable timeout.",
)


def _parse_datasets(datasets: str | Iterable[str] | None) -> list[str]:
    if datasets is None:
        return []
    if isinstance(datasets, str):
        return [x.strip() for x in datasets.split(",") if x.strip()]
    return [str(x).strip() for x in datasets if str(x).strip()]


def _has_key(cfg: DictConfig, key: str) -> bool:
    return isinstance(cfg, DictConfig) and key in cfg


def _apply_eval_overrides(_C: DictConfig, _A: argparse.Namespace) -> DictConfig:
    """Apply CLI overrides to an evaluation config object in-place."""
    if not _has_key(_C, "evaluator"):
        return _C

    evaluator_cfg = _C.evaluator

    requested_datasets = _parse_datasets(_A.datasets)
    if requested_datasets:
        if not _has_key(evaluator_cfg, "datasets_and_prompts"):
            raise ValueError(
                "--datasets was provided, but this evaluator config does not have "
                "a datasets_and_prompts field. Use it with zero-shot classification configs."
            )

        available = list(evaluator_cfg.datasets_and_prompts.keys())
        missing = [d for d in requested_datasets if d not in available]
        if missing:
            raise ValueError(
                f"Requested dataset(s) {missing} are not available in this config. "
                f"Available datasets: {available}"
            )

        evaluator_cfg.datasets_and_prompts = {
            d: evaluator_cfg.datasets_and_prompts[d] for d in requested_datasets
        }
        logger.info(f"Filtered evaluation datasets to: {requested_datasets}")

    if _A.eval_data_dir:
        evaluator_cfg.data_dir = _A.eval_data_dir
        logger.info(f"Overrode evaluator.data_dir to: {_A.eval_data_dir}")

    if _A.eval_batch_size is not None:
        evaluator_cfg.batch_size = int(_A.eval_batch_size)
        logger.info(f"Overrode evaluator.batch_size to: {_A.eval_batch_size}")

    if _A.eval_workers is not None:
        evaluator_cfg.num_workers = int(_A.eval_workers)
        logger.info(f"Overrode evaluator.num_workers to: {_A.eval_workers}")

    if _A.eval_timeout is not None:
        evaluator_cfg.timeout = int(_A.eval_timeout)
        logger.info(f"Overrode evaluator.timeout to: {_A.eval_timeout}")

    return _C


def main(_A: argparse.Namespace):
    device = (
        torch.device(f"cuda:{torch.cuda.current_device()}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # Create evaluation and training config objects.
    _C_TRAIN = LazyConfig.load(_A.train_config)

    # If no config is provided, run all evaluations.
    if _A.config is None:
        config_paths = [
            "configs/eval_zero_shot_classification.py",
            "configs/eval_zero_shot_retrieval.py",
            "configs/eval_hierarchical_metrics.py",
        ]

        all_results = {}
        for config_path in config_paths:
            logger.info(f"Running evaluation with config: {config_path}")
            _C = LazyConfig.load(config_path)
            _C = _apply_eval_overrides(_C, _A)
            logger.info(OmegaConf.to_yaml(_C))

            # Get evaluation task name for results key.
            eval_task_name = (
                os.path.basename(config_path).replace("eval_", "").replace(".py", "")
            )

            result = run_single_evaluation(_A, _C, _C_TRAIN, device, eval_task_name)
            save_individual_result(_A, _C, result, eval_task_name)
            all_results.update(result)

        output_combined_results(_A, all_results)

    else:
        _C = LazyConfig.load(_A.config)
        _C = _apply_eval_overrides(_C, _A)
        logger.info(OmegaConf.to_yaml(_C))
        eval_task_name = (
            os.path.basename(_A.config).replace("eval_", "").replace(".py", "")
        )
        result = run_single_evaluation(_A, _C, _C_TRAIN, device, eval_task_name)
        save_individual_result(_A, _C, result, eval_task_name)
        output_combined_results(_A, result)


def run_single_evaluation(_A, _C, _C_TRAIN, device, eval_task_name):
    logger.info("Command line args:")
    for arg in vars(_A):
        logger.info(f"{arg:<20}: {getattr(_A, arg)}")

    logger.info(f"Evaluating checkpoint in {_A.checkpoint_path}...")

    # Create a fresh model and evaluator for every checkpoint, so the evaluator
    # is free to modify the model weights (e.g. remove projection layers).
    evaluator = instantiate(_C.evaluator)
    model = LazyFactory.build_model(_C_TRAIN, device).eval()

    # Explicitly move model to device.
    model = model.to(device)
    logger.info(f"Model moved to device: {next(model.parameters()).device}")

    CheckpointManager(model=model).load(_A.checkpoint_path)

    results_dict = evaluator(model)

    # Convert Tensor types to serializable format using item().
    def to_serializable(val):
        if isinstance(val, torch.Tensor) and val.numel() == 1:
            return val.item()
        if isinstance(val, dict):
            return {k: to_serializable(v) for k, v in val.items()}
        return val

    results_dict = {k: to_serializable(v) for k, v in results_dict.items()}

    # Add evaluation task prefix to results.
    prefixed_results = {f"{eval_task_name}_{k}": v for k, v in results_dict.items()}

    return prefixed_results


def save_individual_result(_A, _C, results_dict, eval_task_name):
    # Remove prefix from results for individual saving.
    original_results = {}
    prefix = f"{eval_task_name}_"
    for k, v in results_dict.items():
        if k.startswith(prefix):
            original_results[k[len(prefix) :]] = v
        else:
            original_results[k] = v

    # Additional information: evaluation task name.
    eval_task = None
    if hasattr(_C.evaluator, "_target_"):
        eval_task = _C.evaluator._target_

    save_obj = {
        "eval_task": eval_task,
        "checkpoint_path": _A.checkpoint_path,
        "config_path": _A.config,
        "train_config": _A.train_config,
        "datasets": _parse_datasets(_A.datasets),
        "eval_data_dir": _A.eval_data_dir,
        "results": original_results,
    }

    # Save results as JSON in the experiment directory (append as list).
    exp_dir = os.path.dirname(os.path.abspath(_A.checkpoint_path))
    json_path = os.path.join(exp_dir, "evaluate.json")
    try:
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                try:
                    existing = json.load(f)
                except Exception:
                    existing = []
            if isinstance(existing, dict):
                existing = [existing]
            elif not isinstance(existing, list):
                existing = []
            existing.append(save_obj)
            with open(json_path, "w") as f:
                json.dump(existing, f, indent=2, ensure_ascii=False)
        else:
            with open(json_path, "w") as f:
                json.dump([save_obj], f, indent=2, ensure_ascii=False)
        logger.info(f"Saved evaluation results to {json_path}")
    except Exception as e:
        logger.error(f"Failed to save evaluation results to {json_path}: {e}")


def output_combined_results(_A, results_dict):
    # If any value is dict type, treat as multi-dataset/metrics format.
    if any(isinstance(v, dict) for v in results_dict.values()):
        # Collect metric names from dict values.
        all_metrics = set()
        for v in results_dict.values():
            if isinstance(v, dict):
                all_metrics.update(v.keys())
        all_metrics = sorted(all_metrics)
        header = "dataset," + ",".join(all_metrics)
        lines = []
        for dset, metrics in results_dict.items():
            row = [dset]
            if isinstance(metrics, dict):
                for m in all_metrics:
                    val = metrics.get(m, float("nan"))
                    row.append(f"{val:.3f}" if isinstance(val, float) else str(val))
            else:
                # If not dict type, use as is.
                row += [str(metrics)] + ["" for _ in range(len(all_metrics) - 1)]
            lines.append(",".join(row))
        numbers = "\n".join(lines)
    else:
        header = ",".join(results_dict.keys())
        numbers = ",".join(
            [
                f"{num:.3f}" if isinstance(num, float) else str(num)
                for num in results_dict.values()
            ]
        )

    logger.info(f"copypaste: {_A.checkpoint_path}")
    logger.info(f"\ncopypaste below:\n{header}\n{numbers}")


if __name__ == "__main__":
    _A = parser.parse_args()
    main(_A)
