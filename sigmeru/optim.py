# ---------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------

from __future__ import annotations

import math

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer


class LinearWarmupCosineDecayLR(LambdaLR):
    """
    A learning rate scheduler which linearly increases learning rate from 0
    LR, and further decreases it to zero by cosine decay.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int,
        last_epoch: int = -1,
    ):
        assert warmup_steps < total_steps, (
            "Warmup steps should be less than total steps."
        )

        self.tsteps = total_steps
        self.wsteps = warmup_steps
        super().__init__(optimizer, self._lr_multiplier, last_epoch)

    def _lr_multiplier(self, step: int) -> float:
        if step < self.wsteps:
            multiplier = step / float(max(1, self.wsteps))
        else:
            cos_factor = (step - self.wsteps) / (self.tsteps - self.wsteps)
            multiplier = math.cos(cos_factor * (math.pi / 2)) ** 2
        return max(0, multiplier)


def set_weight_decay_per_param(
    model: torch.nn.Module,
    weight_decay: float,
    gain_bias_decay: float | None = None,
    exclude_params: list[str] = [],
) -> list[dict]:
    """
    Set weight decay for trainable parameters of a model.

    ``exclude_params`` supports substring matching against both the local
    parameter name and the full dotted parameter name. This keeps the old names
    such as ``curv`` working, and also correctly handles ``ParameterList`` names
    such as ``curvs.0`` or ``visual_concept_alphas.3`` in PHyCLIP.
    """
    norm_classes = (
        torch.nn.modules.batchnorm._BatchNorm,
        torch.nn.LayerNorm,
        torch.nn.GroupNorm,
        torch.nn.modules.instancenorm._InstanceNorm,
        torch.nn.LocalResponseNorm,
    )

    gain_bias_decay = gain_bias_decay or weight_decay
    params = {"regular": [], "gain_bias": [], "excluded": []}
    params_weight_decay = {
        "regular": weight_decay,
        "gain_bias": gain_bias_decay,
        "excluded": 0.0,
    }

    already_added_parameters = set()

    def _is_excluded(local_name: str, full_name: str) -> bool:
        return any(
            exclude_name in local_name or exclude_name in full_name
            for exclude_name in exclude_params
        )

    def _add_params(module, prefix=""):
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad or p in already_added_parameters:
                continue

            already_added_parameters.add(p)
            full_name = f"{prefix}.{name}" if prefix else name

            if _is_excluded(name, full_name):
                params["excluded"].append(p)
            elif isinstance(module, norm_classes) or "bias" in name:
                params["gain_bias"].append(p)
            else:
                params["regular"].append(p)

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            _add_params(child_module, prefix=child_prefix)

    _add_params(model)

    param_groups = []
    for key in params:
        if len(params[key]) > 0:
            param_groups.append(
                {"params": params[key], "weight_decay": params_weight_decay[key]}
            )
    return param_groups
