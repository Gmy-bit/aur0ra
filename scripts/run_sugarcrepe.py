#!/usr/bin/env python3
"""Evaluate SugarCrepe using the prepared JSONL data and the shared evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sigmeru.config import LazyConfig, LazyFactory
from sigmeru.evaluation.scene_understanding import SceneUnderstandingEvaluator
from sigmeru.utils.checkpointing import CheckpointManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pairwise SugarCrepe evaluation for a trained model."
    )
    parser.add_argument("--train-config", required=True, help="Training config (.py or saved config.yaml).")
    parser.add_argument("--checkpoint-path", required=True, help="Model checkpoint path.")
    parser.add_argument("--eval-data-dir", default="datasets/eval", help="Evaluation data root.")
    parser.add_argument("--device", default="cuda:0", help="Inference device.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--annotation-file",
        default="scene_understanding/sugarcrepe.jsonl",
        help="JSONL path relative to --eval-data-dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_cfg = LazyConfig.load(args.train_config)
    model = LazyFactory.build_model(train_cfg, device).eval().to(device)
    CheckpointManager(model=model).load(args.checkpoint_path)

    evaluator = SceneUnderstandingEvaluator(
        data_dir=args.eval_data_dir,
        benchmarks={
            "SugarCrepe": {
                "annotation_file": args.annotation_file,
                "image_root": ".",
                "group_key": "group",
            }
        },
        image_size=224,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        timeout=30,
        max_samples=None,
        tie_is_correct=False,
        include_overall=True,
    )
    results = evaluator(model)

    checkpoint_dir = Path(args.checkpoint_path).expanduser().resolve().parent
    output_dir = checkpoint_dir / "sugarcrepe_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "metrics.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Saved metrics to: {output_path}")


if __name__ == "__main__":
    main()
