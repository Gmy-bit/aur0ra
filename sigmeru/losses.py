from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

_VALID_LOSSES = {"contrastive", "sigmoid"}
_VALID_SIGMOID_NORMALIZATIONS = {"batch", "pairs"}
_VALID_MAIN_SCORE_TYPES = {"lorentz_distance", "rapidity", "root_angle", "root_angle_cos", "root_angle_cosine"}
_VALID_ENTAILMENT_LOSS_TYPES = {
    "none",
    "positive_cos",
    "sigmoid_cos",
    "contrastive_cos",
    "typed_shadow",
    "radial_shadow",
    "radial_shadow_decomposed",
}
_VALID_NORM_REG_TYPES = {
    "none",
    "argent_space",
    "argent_radius",
    "inverse",
    "inverse_space",
    "inverse_square",
    "inverse_square_space",
    "softplus_neg",
    "softplus_neg_space",
    "inverse_radius",
    "inverse_square_radius",
    "softplus_neg_radius",
}
_VALID_ANGULAR_REDUCTIONS = {"pair_mean", "balanced"}
_VALID_ANGULAR_NEGATIVE_MINING = {"all", "topk"}


def validate_loss_type(loss_type: str) -> str:
    loss_type = str(loss_type).lower().strip()
    if loss_type not in _VALID_LOSSES:
        raise ValueError(f"loss_type must be one of {sorted(_VALID_LOSSES)}, got {loss_type!r}.")
    return loss_type


def validate_sigmoid_normalization(sigmoid_normalization: str) -> str:
    sigmoid_normalization = str(sigmoid_normalization).lower().strip()
    if sigmoid_normalization not in _VALID_SIGMOID_NORMALIZATIONS:
        raise ValueError(
            "sigmoid_normalization must be one of "
            f"{sorted(_VALID_SIGMOID_NORMALIZATIONS)}, got {sigmoid_normalization!r}."
        )
    return sigmoid_normalization


def validate_main_score_type(main_score_type: str) -> str:
    main_score_type = str(main_score_type).lower().strip()
    if main_score_type not in _VALID_MAIN_SCORE_TYPES:
        raise ValueError(
            f"main_score_type must be one of {sorted(_VALID_MAIN_SCORE_TYPES)}, got {main_score_type!r}."
        )
    # Public alias: the training score is cos(root_angle), but the mode name is
    # kept short because this is the main root-angle objective.
    if main_score_type in {"root_angle_cos", "root_angle_cosine"}:
        return "root_angle"
    return main_score_type


def validate_entailment_loss_type(entailment_loss_type: str) -> str:
    entailment_loss_type = str(entailment_loss_type).lower().strip()
    entailment_loss_type = {
        "cone_depth": "typed_shadow",
        "shadow_depth": "typed_shadow",
        "typed_boundary_shadow": "typed_shadow",
        "typed_boundary_shadow_cone": "typed_shadow",
        "typed_boundary_shadow_cones": "typed_shadow",
        "mtrs": "radial_shadow",
        "radial_shadow_order": "radial_shadow",
        "modality_typed_radial_shadow": "radial_shadow",
        "modality_typed_radial_shadow_order": "radial_shadow",
        "radial_shadow_split": "radial_shadow_decomposed",
        "mtrs_decomposed": "radial_shadow_decomposed",
        "modality_typed_radial_shadow_decomposed": "radial_shadow_decomposed",
        "modality_typed_radial_shadow_split": "radial_shadow_decomposed",
    }.get(entailment_loss_type, entailment_loss_type)
    if entailment_loss_type not in _VALID_ENTAILMENT_LOSS_TYPES:
        raise ValueError(
            "entailment_loss_type must be one of "
            f"{sorted(_VALID_ENTAILMENT_LOSS_TYPES)}, got {entailment_loss_type!r}."
        )
    return entailment_loss_type


def validate_norm_reg_type(norm_reg_type: str) -> str:
    norm_reg_type = str(norm_reg_type).lower().strip()
    norm_reg_type = {
        "inverse": "inverse_space",
        "inverse_square": "inverse_square_space",
        "softplus_neg": "softplus_neg_space",
    }.get(norm_reg_type, norm_reg_type)
    if norm_reg_type not in _VALID_NORM_REG_TYPES:
        raise ValueError(
            f"norm_reg_type must be one of {sorted(_VALID_NORM_REG_TYPES)}, got {norm_reg_type!r}."
        )
    return norm_reg_type


def validate_angular_reduction(angular_reduction: str) -> str:
    angular_reduction = str(angular_reduction).lower().strip()
    if angular_reduction not in _VALID_ANGULAR_REDUCTIONS:
        raise ValueError(
            "angular_reduction must be one of "
            f"{sorted(_VALID_ANGULAR_REDUCTIONS)}, got {angular_reduction!r}."
        )
    return angular_reduction


def validate_angular_negative_mining(angular_negative_mining: str) -> str:
    angular_negative_mining = str(angular_negative_mining).lower().strip()
    if angular_negative_mining not in _VALID_ANGULAR_NEGATIVE_MINING:
        raise ValueError(
            "angular_negative_mining must be one of "
            f"{sorted(_VALID_ANGULAR_NEGATIVE_MINING)}, got {angular_negative_mining!r}."
        )
    return angular_negative_mining


def shifted_targets(batch_size: int, rank: int, device: torch.device) -> Tensor:
    """Return global positive indices for local batch rows."""
    return torch.arange(batch_size, device=device) + batch_size * rank


def symmetric_cross_entropy_loss(
    image_logits: Tensor,
    text_logits: Tensor,
    targets: Tensor,
    scale: Tensor | float,
) -> Tensor:
    """CLIP/MERU-style symmetric softmax contrastive objective."""
    return 0.5 * (
        F.cross_entropy(scale * image_logits, targets)
        + F.cross_entropy(scale * text_logits, targets)
    )


def _sample_negative_columns(
    num_rows: int,
    num_cols: int,
    targets: Tensor,
    negatives_per_sample: int,
    device: torch.device,
) -> Tensor:
    if negatives_per_sample <= 0 or negatives_per_sample >= num_cols - 1:
        raise ValueError("_sample_negative_columns is only for finite negative sampling.")
    raw = torch.randint(
        low=0,
        high=num_cols - 1,
        size=(num_rows, negatives_per_sample),
        device=device,
    )
    return raw + (raw >= targets[:, None]).to(raw.dtype)


def sigmoid_loss_from_logits(
    logits: Tensor,
    targets: Tensor,
    normalization: str = "batch",
    negatives_per_sample: int = 0,
) -> Tensor:
    """
    Dense-label-free pairwise SigLIP loss.

    Positives are at ``targets`` and all other columns are negatives.  The full
    negative path avoids explicitly allocating a dense +/-1 label matrix.
    """
    normalization = validate_sigmoid_normalization(normalization)
    num_rows, num_cols = logits.shape
    row_ids = torch.arange(num_rows, device=logits.device)
    positive_logits = logits[row_ids, targets]

    if 0 < negatives_per_sample < num_cols - 1:
        neg_cols = _sample_negative_columns(num_rows, num_cols, targets, negatives_per_sample, logits.device)
        neg_logits = logits.gather(dim=1, index=neg_cols)
        loss_sum = F.softplus(neg_logits).sum() + F.softplus(-positive_logits).sum()
        denom_pairs = num_rows * (negatives_per_sample + 1)
    else:
        # Sum softplus(logit_ij) for all pairs, then replace positive terms
        # softplus(pos) by softplus(-pos) = softplus(pos) - pos.
        loss_sum = F.softplus(logits).sum() - positive_logits.sum()
        denom_pairs = logits.numel()

    if normalization == "batch":
        return loss_sum / num_rows
    return loss_sum / denom_pairs


def symmetric_sigmoid_loss_from_logits(
    image_logits: Tensor,
    text_logits: Tensor,
    targets: Tensor,
    normalization: str = "batch",
    negatives_per_sample: int = 0,
) -> Tensor:
    """Symmetric image->text and text->image pairwise sigmoid objective."""
    return 0.5 * (
        sigmoid_loss_from_logits(image_logits, targets, normalization, negatives_per_sample)
        + sigmoid_loss_from_logits(text_logits, targets, normalization, negatives_per_sample)
    )


def image_text_loss(
    loss_type: str,
    image_logits: Tensor,
    text_logits: Tensor,
    targets: Tensor,
    scale: Tensor | float,
    bias: Tensor | float | None = None,
    sigmoid_normalization: str = "batch",
    sigmoid_negatives_per_sample: int = 0,
) -> Tensor:
    """
    Dispatch image-text alignment loss.

    ``image_logits`` and ``text_logits`` are *scores* for contrastive loss. For
    sigmoid loss this function applies ``scale * score + bias`` before the
    pairwise binary objective.
    """
    loss_type = validate_loss_type(loss_type)
    if loss_type == "contrastive":
        return symmetric_cross_entropy_loss(image_logits, text_logits, targets, scale)

    image_pair_logits = scale * image_logits
    text_pair_logits = scale * text_logits
    if bias is not None:
        image_pair_logits = image_pair_logits + bias
        text_pair_logits = text_pair_logits + bias
    return symmetric_sigmoid_loss_from_logits(
        image_pair_logits,
        text_pair_logits,
        targets=targets,
        normalization=sigmoid_normalization,
        negatives_per_sample=sigmoid_negatives_per_sample,
    )
