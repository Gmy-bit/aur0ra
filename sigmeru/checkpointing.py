# ---------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------

from __future__ import annotations

import copy
from pathlib import Path
from typing import Iterable

import torch
from loguru import logger
from torch.nn.parallel import DistributedDataParallel

import sigmeru.utils.distributed as dist


# -----------------------------------------------------------------------------
# Backward-compatibility keys introduced after early MERU / SigMERU checkpoints.
#
# These keys are safe to leave initialized from the current config when loading
# older checkpoints for evaluation. They are either:
#   1) parameters/buffers used only by newer experimental modes, or
#   2) anchor values for optional regularization.
#
# We do NOT globally use strict=False, because that would hide real architecture
# mismatches. Only these known compatibility keys are ignored.
# -----------------------------------------------------------------------------
_MODEL_COMPAT_MISSING_KEYS = {
    # v15 tangent-radius ablations.
    "visual_tangent_log_radius",
    "textual_tangent_log_radius",

    # v16/v19 omega-scaled Lorentz inner product.
    "sigmoid_inner_log_omega",

    # v14/v15 distance affine mapping.
    "sigmoid_distance_affine_center",
    "sigmoid_distance_affine_log_scale",

    # v14/v15 anchor regularization buffers.
    "_curv_init_log",
    "_alpha_init_log",
    "_tangent_radius_init_log",

    # v13/v14/v15 distance EMA buffers.
    "sigmoid_distance_ema_mean",
    "sigmoid_distance_ema_std",
    "sigmoid_distance_ema_initialized",
}

_MODEL_COMPAT_MISSING_PREFIXES = (
    # Optional tangent-space BatchNorm modules, present only in some newer configs.
    "visual_tangent_bn.",
    "textual_tangent_bn.",
)


def _unwrap_ddp(obj):
    return obj.module if isinstance(obj, DistributedDataParallel) else obj


def _is_compat_missing_key(key: str) -> bool:
    if key in _MODEL_COMPAT_MISSING_KEYS:
        return True
    return any(key.startswith(prefix) for prefix in _MODEL_COMPAT_MISSING_PREFIXES)


def _format_key_list(keys: Iterable[str], max_items: int = 40) -> str:
    keys = list(keys)
    if not keys:
        return "[]"
    if len(keys) <= max_items:
        return "[" + ", ".join(keys) + "]"
    head = keys[:max_items]
    return "[" + ", ".join(head) + f", ... +{len(keys) - max_items} more]"


class CheckpointManager:
    """
    Utility class to periodically save PyTorch models and other checkpointables
    such as optimizers, LR schedulers, and AMP gradient scalers.

    For PyTorch DistributedDataParallel objects, the internal module state_dict
    is saved and loaded.
    """

    def __init__(
        self,
        output_dir: str | Path = "/tmp",
        keep_recent: int = 100,
        **checkpointables,
    ):
        """
        Args:
            output_dir: Path to a directory to save checkpoints.
            keep_recent: Number of recent checkpoints to keep.
            checkpointables: Keyword arguments with checkpointable objects,
                for example model, optimizer, scheduler, scaler.
        """
        self.output_dir = Path(output_dir)
        self.keep_recent = keep_recent
        self._recent_iterations: list[str] = []

        # Shallow copy keeps references to original objects.
        self.checkpointables = copy.copy(checkpointables)

    def step(self, iteration: int):
        """
        Save a checkpoint.

        Args:
            iteration: Current training iteration.
        """
        out_state_dict = {}
        for key, obj in self.checkpointables.items():
            out_state_dict[key] = _unwrap_ddp(obj).state_dict()

        out_state_dict["iteration"] = iteration

        iter_str = f"{iteration:0>8d}"
        ckpt_path = self.output_dir / f"checkpoint_{iter_str}.pth"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(out_state_dict, ckpt_path)

        with (self.output_dir / "last_checkpoint.txt").open("w") as f:
            f.write(f"checkpoint_{iter_str}.pth")

        self._recent_iterations.append(iter_str)
        if len(self._recent_iterations) > self.keep_recent:
            oldest_iteration = self._recent_iterations.pop(0)
            oldest_path = self.output_dir / f"checkpoint_{oldest_iteration}.pth"
            if oldest_path.exists():
                oldest_path.unlink()

    def final_step(self):
        """
        Save the final checkpoint with name checkpoint_final.pth.
        This method does not update last_checkpoint.txt.
        """
        out_state_dict = {}
        for key, obj in self.checkpointables.items():
            out_state_dict[key] = _unwrap_ddp(obj).state_dict()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(out_state_dict, self.output_dir / "checkpoint_final.pth")

    def resume(self) -> int:
        """
        Find and load the last saved checkpoint in output_dir.

        Returns:
            Iteration corresponding to the loaded checkpoint. Returns 0 if no
            checkpoint is found.
        """
        logger.info(f"Attempting to resume job from {self.output_dir}...")

        last_ckpt_info_file = self.output_dir / "last_checkpoint.txt"
        if last_ckpt_info_file.exists():
            ckpt_path = last_ckpt_info_file.read_text().strip()
            logger.info(f"Found last checkpoint in {self.output_dir}: {ckpt_path}")
            return self.load(self.output_dir / ckpt_path)

        logger.warning(
            f"No checkpoint found in {self.output_dir} to resume job. "
            "Hopefully this is the beginning of a fresh job."
        )
        return 0

    def _load_model_state_dict_compat(self, model, state_dict: dict, path: str | Path):
        """
        Load model state dict with strict compatibility for known newly-added keys.

        This is mainly for evaluating older checkpoints after adding new optional
        SigMERU experiment parameters/buffers.
        """
        model = _unwrap_ddp(model)
        result = model.load_state_dict(state_dict, strict=False)

        missing_keys = list(result.missing_keys)
        unexpected_keys = list(result.unexpected_keys)

        bad_missing = [key for key in missing_keys if not _is_compat_missing_key(key)]
        bad_unexpected = unexpected_keys

        if bad_missing or bad_unexpected:
            message_parts = [f"Error(s) in loading model state_dict from {path}."]
            if bad_missing:
                message_parts.append(
                    "Unexpected missing key(s): "
                    + _format_key_list(bad_missing)
                )
            if bad_unexpected:
                message_parts.append(
                    "Unexpected extra key(s): "
                    + _format_key_list(bad_unexpected)
                )
            if missing_keys:
                message_parts.append(
                    "All missing key(s): "
                    + _format_key_list(missing_keys)
                )
            raise RuntimeError("\n".join(message_parts))

        if missing_keys:
            logger.warning(
                "Loaded model with backward-compatible missing key(s): "
                f"{_format_key_list(missing_keys)}. "
                "These are newly-added optional SigMERU parameters/buffers and "
                "will keep their current initialized values."
            )

    def _load_state_dict_for_checkpointable(
        self,
        key: str,
        obj,
        state_dict: dict,
        path: str | Path,
    ):
        """
        Load state dict for one checkpointable.

        Model loading is backward-compatible for known newly added keys.
        Optimizer/scheduler/scaler loading remains strict.
        """
        if key == "model":
            self._load_model_state_dict_compat(obj, state_dict, path)
        else:
            _unwrap_ddp(obj).load_state_dict(state_dict)

    def load(self, path: str | Path) -> int:
        """
        Load a saved checkpoint from path.

        Args:
            path: Path to a checkpoint saved by step/final_step.

        Returns:
            Iteration corresponding to the loaded checkpoint. If iteration is
            not found in the file, returns -1.
        """
        rank = dist.get_rank()
        path = Path(path)

        logger.info(f"Rank {rank}: Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        iteration = checkpoint.pop("iteration", -1)

        is_loaded = {key: False for key in self.checkpointables}

        for key, state_dict in checkpoint.items():
            if key not in self.checkpointables:
                logger.info(f"Rank {rank}: {key} not found in checkpointables.")
                continue

            logger.info(f"Rank {rank}: Loading {key} from {path}")
            self._load_state_dict_for_checkpointable(
                key=key,
                obj=self.checkpointables[key],
                state_dict=state_dict,
                path=path,
            )
            is_loaded[key] = True

        not_loaded = [key for key, loaded in is_loaded.items() if not loaded]
        if not_loaded:
            logger.info(
                f"Rank {rank}: Checkpointables not found in file: {not_loaded}"
            )

        logger.info(f"Rank {rank}: Loaded checkpoint iteration {iteration}")
        return iteration