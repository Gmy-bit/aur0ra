# ---------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------

# Modified from github.com/facebookresearch/meru

from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from sigmeru import lorentz as L
from sigmeru.encoders.text_encoders import TransformerTextEncoder
from sigmeru.losses import (
    image_text_loss,
    shifted_targets,
    sigmoid_loss_from_logits,
    validate_angular_negative_mining,
    validate_angular_reduction,
    validate_entailment_loss_type,
    validate_loss_type,
    validate_main_score_type,
    validate_norm_reg_type,
    validate_sigmoid_normalization,
)
import sigmeru.utils.distributed as dist


# -----------------------------------------------------------------------------
# Numerically safe Lorentz helpers.
# The project stores Lorentz points by their spatial components only; the time
# component is reconstructed from curvature when needed.
# -----------------------------------------------------------------------------

def lorentz_time(x: torch.Tensor, curv: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.clamp(1.0 / curv + torch.sum(x * x, dim=-1, keepdim=True), min=1e-30))


def pairwise_rapidity(
    x: torch.Tensor,
    y: torch.Tensor,
    curv: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Pairwise curvature-normalized rapidity rho=acosh(-c <x,y>_L)."""
    c_xy = -curv * L.pairwise_inner(x, y, curv)
    return torch.acosh(torch.clamp(c_xy, min=1.0 + eps))


def pairwise_lorentz_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    curv: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    return pairwise_rapidity(x, y, curv, eps=eps) / torch.sqrt(curv)


def radius_from_origin(x: torch.Tensor, curv: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Geodesic distance from the Lorentz origin to x."""
    x_time_scaled = torch.sqrt(torch.clamp(1.0 + curv * torch.sum(x * x, dim=-1), min=1.0 + eps))
    return torch.acosh(torch.clamp(x_time_scaled, min=1.0 + eps)) / torch.sqrt(curv)


def normalized_radius_from_origin(x: torch.Tensor, curv: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """sqrt(c) times the geodesic radius; equals rapidity from origin."""
    x_time_scaled = torch.sqrt(torch.clamp(1.0 + curv * torch.sum(x * x, dim=-1), min=1.0 + eps))
    return torch.acosh(torch.clamp(x_time_scaled, min=1.0 + eps))


def pairwise_root_angle_cos(
    x: torch.Tensor,
    y: torch.Tensor,
    curv: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Pairwise cos of the root angle ∠xOy at the Lorentz origin O.

    In the Lorentz model with points represented by spatial components, the
    geodesic from O to a point preserves the spatial ray direction. Therefore
    this is exactly the cosine between log_O(x) and log_O(y), and is also what
    the hyperbolic law of cosines gives for the triangle x-O-y.  The curvature
    argument is accepted for API symmetry, but cancels out of the final formula.
    """
    del curv
    if x.dim() != 2 or y.dim() != 2:
        raise ValueError(
            "pairwise_root_angle_cos expects two 2D tensors with shapes "
            f"[num_x, dim] and [num_y, dim], got {tuple(x.shape)} and {tuple(y.shape)}."
        )
    x_dir = F.normalize(x, dim=-1, eps=eps)
    y_dir = F.normalize(y, dim=-1, eps=eps)
    cos_theta = x_dir @ y_dir.transpose(0, 1)
    return torch.clamp(cos_theta, min=-1.0 + eps, max=1.0 - eps)


def pairwise_root_angle(
    x: torch.Tensor,
    y: torch.Tensor,
    curv: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    return torch.acos(pairwise_root_angle_cos(x, y, curv=curv, eps=eps))


def typed_boundary_shadow_depth(
    parent_text: torch.Tensor,
    child_image: torch.Tensor,
    curv: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Signed depth for typed boundary shadow cones.

    Text is modeled as a semantic region with aperture
    psi(t)=asin(sech(rho_t)); image is modeled as a leaf with zero aperture.
    The returned depth is psi(t)-theta(t,v), where theta is the root angle
    between text and image spatial directions.  Positive values are inside the
    text cone; larger values are deeper inside.
    """
    theta_cos = pairwise_root_angle_cos(parent_text, child_image, curv=curv, eps=eps)
    # 1e-8 is below float32 resolution around 1.0.  Use a representable
    # margin before acos so nearly identical directions cannot create infinite
    # gradients under AMP/float32.
    angle_eps = max(float(eps), 1e-6)
    theta = torch.acos(theta_cos.clamp(-1.0 + angle_eps, 1.0 - angle_eps))
    rho_text = normalized_radius_from_origin(parent_text, curv, eps=eps)
    trig_eps = max(float(eps), 1e-6)
    aperture = torch.asin(torch.cosh(rho_text).reciprocal().clamp(trig_eps, 1.0 - trig_eps))
    depth = aperture[:, None] - theta
    return depth, theta, aperture


def modality_typed_radial_shadow_depth(
    parent_text: torch.Tensor,
    child_image: torch.Tensor,
    curv: torch.Tensor,
    radial_slope: float = 8.0,
    radial_scale: float = 0.1,
    eps: float = 1e-8,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Signed depth for the Modality-Typed Radial Shadow order.

    Text is the semantic parent and image is the typed leaf.  A valid pair must
    satisfy two constraints simultaneously:

      1) radial order: rho_image >= rho_text;
      2) shadow order: theta(text,image) <= slope * (rho_image-rho_text)_+.

    Here rho=sqrt(c)*d_H(O,x) is the curvature-normalized radius and theta is
    the root angle.  The allowed angle is clipped at pi only because spherical
    angular distance cannot exceed pi; this does not alter the order once the
    shadow covers the full boundary sphere.

    For stable sigmoid optimization, the two signed margins are normalized and
    clipped to [-1,1], and the joint depth is their pointwise minimum.  Hence a
    positive depth requires both radial and angular constraints to hold.
    """
    if radial_slope <= 0:
        raise ValueError("radial_slope must be positive.")
    if radial_scale <= 0:
        raise ValueError("radial_scale must be positive.")

    theta_cos = pairwise_root_angle_cos(parent_text, child_image, curv=curv, eps=eps)
    angle_eps = max(float(eps), 1e-6)
    theta = torch.acos(theta_cos.clamp(-1.0 + angle_eps, 1.0 - angle_eps))
    rho_text = normalized_radius_from_origin(parent_text, curv, eps=eps)
    rho_image = normalized_radius_from_origin(child_image, curv, eps=eps)
    radial_gap = rho_image.unsqueeze(0) - rho_text.unsqueeze(1)

    allowed_angle = (float(radial_slope) * radial_gap.clamp_min(0.0)).clamp_max(math.pi)
    radial_margin = (radial_gap / float(radial_scale)).clamp(-1.0, 1.0)
    angular_margin = ((allowed_angle - theta) / math.pi).clamp(-1.0, 1.0)
    depth = torch.minimum(radial_margin, angular_margin)
    return depth, theta, allowed_angle, radial_gap, rho_text, rho_image


def compute_paired_exterior_angle_cos(
    parent_text: torch.Tensor,
    child_image: torch.Tensor,
    curv: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Elementwise cos of MERU exterior angle for aligned text-image pairs."""
    x = parent_text
    y = child_image
    x_time = lorentz_time(x, curv).squeeze(-1)
    y_time = lorentz_time(y, curv).squeeze(-1)
    c_xyl = curv * (-x_time * y_time + torch.sum(x * y, dim=-1))
    numer = y_time + c_xyl * x_time
    denom_inner = torch.clamp(c_xyl * c_xyl - 1.0, min=eps)
    x_space_norm = torch.norm(x, dim=-1).clamp_min(eps)
    denom = x_space_norm * torch.sqrt(denom_inner)
    cos_alpha = numer / denom.clamp_min(eps)
    return torch.clamp(cos_alpha, min=-1.0 + eps, max=1.0 - eps)


def compute_exterior_angle_cos(
    parent_text: torch.Tensor,
    child_image: torch.Tensor,
    curv: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Pairwise cos of exterior angle alpha(text_parent, image_child).

    This is the arccos input from the MERU exterior-angle formula, but returned
    directly so training can avoid arccos.  Shape is [num_text, num_image].
    """
    x = parent_text
    y = child_image
    x_time = lorentz_time(x, curv)               # [Bt, 1]
    y_time = lorentz_time(y, curv)               # [Bi, 1]
    c_xyl = curv * L.pairwise_inner(x, y, curv)  # negative, <= -1 ideally
    numer = y_time.T + c_xyl * x_time            # [Bt, Bi]
    denom_inner = torch.clamp(c_xyl * c_xyl - 1.0, min=eps)
    x_space_norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(eps)
    denom = x_space_norm * torch.sqrt(denom_inner)
    cos_alpha = numer / denom.clamp_min(eps)
    return torch.clamp(cos_alpha, min=-1.0 + eps, max=1.0 - eps)


def compute_exterior_angle(
    parent_text: torch.Tensor,
    child_image: torch.Tensor,
    curv: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    return torch.acos(compute_exterior_angle_cos(parent_text, child_image, curv, eps=eps))


class CLIPBaseline(nn.Module):
    """CLIP/SigLIP baseline in Euclidean space."""

    def __init__(
        self,
        visual: nn.Module,
        textual: TransformerTextEncoder,
        embed_dim: int,
        pixel_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        pixel_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        use_boxes: bool = False,
        loss_type: str = "contrastive",
        sigmoid_logit_bias_init: float = -6.5,
        sigmoid_normalization: str = "batch",
        sigmoid_negatives_per_sample: int = 0,
        **_: object,
    ):
        super().__init__()
        self.visual = visual
        self.textual = textual
        self.embed_dim = embed_dim
        self.use_boxes = use_boxes
        self.loss_type = validate_loss_type(loss_type)
        self.sigmoid_normalization = validate_sigmoid_normalization(sigmoid_normalization)
        self.sigmoid_negatives_per_sample = int(sigmoid_negatives_per_sample)

        self.visual_proj = nn.Linear(visual.width, embed_dim, bias=False)
        self.textual_proj = nn.Linear(textual.width, embed_dim, bias=False)
        nn.init.normal_(self.visual_proj.weight, std=visual.width**-0.5)
        nn.init.normal_(self.textual_proj.weight, std=textual.width**-0.5)

        self.logit_scale = nn.Parameter(torch.tensor(10.0).log())
        bias = torch.tensor(float(sigmoid_logit_bias_init))
        if self.loss_type == "sigmoid":
            self.logit_bias = nn.Parameter(bias)
        else:
            self.register_buffer("logit_bias", bias)

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1))
        self._rank = dist.get_rank()

    @property
    def device(self) -> torch.device:
        return self.logit_scale.device

    def encode_image_backbone(self, images: torch.Tensor) -> torch.Tensor:
        return self.visual((images - self.pixel_mean) / self.pixel_std)

    def _prepare_tokens(self, tokens: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        if isinstance(tokens, torch.Tensor):
            return tokens.to(self.device)
        for idx, inst_tokens in enumerate(tokens):
            if len(inst_tokens) > self.textual.context_length:
                eot_token = inst_tokens[-1]
                inst_tokens = inst_tokens[: self.textual.context_length]
                inst_tokens[-1] = eot_token
                tokens[idx] = inst_tokens
        tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
        return tokens.to(self.device)

    def encode_text_backbone(self, tokens: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        tokens = self._prepare_tokens(tokens)
        text_feats = self.textual(tokens)
        eos_indices = tokens.argmax(dim=-1)
        batch_idxs = torch.arange(text_feats.shape[0], device=tokens.device)
        return text_feats[batch_idxs, eos_indices]

    def encode_image(self, images: torch.Tensor, project: bool = True) -> torch.Tensor:
        feats = self.visual_proj(self.encode_image_backbone(images))
        return F.normalize(feats, dim=-1) if project else feats

    def encode_text(self, tokens: list[torch.Tensor] | torch.Tensor, project: bool = True) -> torch.Tensor:
        feats = self.textual_proj(self.encode_text_backbone(tokens))
        return F.normalize(feats, dim=-1) if project else feats

    def encode_image_for_eval(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode_image(images, project=True)

    def encode_text_for_eval(self, tokens: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        return self.encode_text(tokens, project=True)

    def build_text_classifier_features(self, tokens: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        feats = self.encode_text(tokens, project=True)
        return F.normalize(feats.mean(dim=0), dim=-1)

    def compute_eval_similarity(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        return image_features @ text_features.T

    def _clamp_logit_scale(self) -> torch.Tensor:
        with torch.no_grad():
            self.logit_scale.clamp_(max=math.log(100.0))
        return self.logit_scale.exp()

    def forward(
        self,
        images: torch.Tensor,
        tokens: list[torch.Tensor] | torch.Tensor,
        iteration: int | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        image_feats = self.encode_image(images, project=True)
        text_feats = self.encode_text(tokens, project=True)
        all_image_feats = torch.cat(dist.gather_across_processes(image_feats), dim=0)
        all_text_feats = torch.cat(dist.gather_across_processes(text_feats), dim=0)
        scale = self._clamp_logit_scale()
        image_scores = image_feats @ all_text_feats.T
        text_scores = text_feats @ all_image_feats.T
        batch_size = image_feats.shape[0]
        targets = shifted_targets(batch_size, self._rank, image_scores.device)
        loss = image_text_loss(
            self.loss_type,
            image_scores,
            text_scores,
            targets=targets,
            scale=scale,
            bias=self.logit_bias if self.loss_type == "sigmoid" else None,
            sigmoid_normalization=self.sigmoid_normalization,
            sigmoid_negatives_per_sample=self.sigmoid_negatives_per_sample,
        )
        row_ids = torch.arange(batch_size, device=image_scores.device)
        pos_score = 0.5 * (image_scores[row_ids, targets].detach().mean() + text_scores[row_ids, targets].detach().mean())
        neg_score = 0.5 * (image_scores.detach().mean() + text_scores.detach().mean())
        logging = {
            "loss_main": loss,
            "alignment_loss": loss,
            f"{self.loss_type}_loss": loss,
            "main_score_pos_mean": pos_score,
            "main_score_neg_mean": neg_score,
            "main_logit_scale": scale,
            "logit_scale": scale,
        }
        if self.loss_type == "sigmoid":
            logging["main_logit_bias"] = self.logit_bias
            logging["logit_bias"] = self.logit_bias
        return {"loss": loss, "logging": logging}


class MERU(CLIPBaseline):
    """
    Clean SigMERU/MERU route.

    encoder feature -> single projection -> learned alpha scaling -> exp_map0
    -> Lorentz hyperbolic embedding.  Main matching defaults to cos(root angle)
    for the hyperbolic triangle x-O-y, with distance/rapidity kept as ablation
    modes. Optional angular entailment is asymmetric text(parent)->image(child)
    and uses cos(exterior angle), not cone aperture, as the trainable score.
    """

    def __init__(
        self,
        visual: nn.Module,
        textual: TransformerTextEncoder,
        embed_dim: int,
        curv_init: float = 1.0,
        learn_curv: bool = True,
        curv_min: float | None = 0.1,
        curv_max: float | None = 10.0,
        entail_weight: float = 0.0,
        use_boxes: bool = False,
        lorentz_eps: float = 1e-8,
        pixel_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        pixel_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        loss_type: str = "sigmoid",
        sigmoid_logit_bias_init: float = -6.5,
        sigmoid_normalization: str = "batch",
        sigmoid_negatives_per_sample: int = 0,
        main_score_type: str = "root_angle",
        eval_score_type: str = "main",
        entailment_loss_type: str = "none",
        angular_temperature_init: float | None = 0.07,
        angular_logit_scale_init: float = 10.0,
        angular_logit_bias_init: float = -6.5,
        angular_reduction: str = "pair_mean",
        angular_negative_mining: str = "all",
        angular_topk: int = 0,
        norm_reg_type: str = "none",
        norm_weight: float = 0.0,
        norm_reg_eps: float = 1e-6,
        norm_reg_alpha: float = 5.0,
        text_feature_dropout: float = 0.0,
        disp_weight: float = 0.0,
        disp_scale: float = 10.0,
        disp_bias: float = -1.0,
        radial_shadow_slope: float = 8.0,
        radial_shadow_radial_scale: float = 0.1,
        radial_shadow_radial_weight: float = 1.0,
        shadow_overlap_weight: float = 0.0,
        shadow_overlap_scale: float = 10.0,
        shadow_overlap_bias: float = -1.0,
        **_: object,
    ):
        super().__init__(
            visual=visual,
            textual=textual,
            embed_dim=embed_dim,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            use_boxes=use_boxes,
            loss_type=loss_type,
            sigmoid_logit_bias_init=sigmoid_logit_bias_init,
            sigmoid_normalization=sigmoid_normalization,
            sigmoid_negatives_per_sample=sigmoid_negatives_per_sample,
        )
        self.main_score_type = validate_main_score_type(main_score_type)
        eval_score_type = str(eval_score_type).lower().strip()
        if eval_score_type == "main":
            self.eval_score_type = "main"
        elif eval_score_type == "angular_cos":
            # Backward-compatible name for MERU exterior-angle cosine evaluation.
            self.eval_score_type = "angular_cos"
        elif eval_score_type in {"typed_shadow", "cone_depth", "shadow_depth", "typed_boundary_shadow"}:
            self.eval_score_type = "typed_shadow"
        elif eval_score_type in {
            "radial_shadow",
            "mtrs",
            "radial_shadow_order",
            "modality_typed_radial_shadow",
            "radial_shadow_decomposed",
            "radial_shadow_split",
            "mtrs_decomposed",
        }:
            # The decomposed branch changes only the training surrogate.
            # Evaluation keeps the joint min-certificate of the same order.
            self.eval_score_type = "radial_shadow"
        else:
            self.eval_score_type = validate_main_score_type(eval_score_type)
        self.entailment_loss_type = validate_entailment_loss_type(entailment_loss_type)
        self.entail_weight = float(entail_weight)
        self.angular_reduction = validate_angular_reduction(angular_reduction)
        self.angular_negative_mining = validate_angular_negative_mining(angular_negative_mining)
        self.angular_topk = int(angular_topk)
        if self.angular_topk < 0:
            raise ValueError("angular_topk must be non-negative.")
        self.norm_reg_type = validate_norm_reg_type(norm_reg_type)
        self.norm_weight = float(norm_weight)
        self.lorentz_eps = float(lorentz_eps)
        self.norm_reg_eps = float(norm_reg_eps)
        self.norm_reg_alpha = float(norm_reg_alpha)
        self.text_feature_dropout = float(text_feature_dropout)
        self.disp_weight = float(disp_weight)
        self.disp_scale = float(disp_scale)
        self.disp_bias = float(disp_bias)
        self.radial_shadow_slope = float(radial_shadow_slope)
        self.radial_shadow_radial_scale = float(radial_shadow_radial_scale)
        self.radial_shadow_radial_weight = float(radial_shadow_radial_weight)
        self.shadow_overlap_weight = float(shadow_overlap_weight)
        self.shadow_overlap_scale = float(shadow_overlap_scale)
        self.shadow_overlap_bias = float(shadow_overlap_bias)
        if self.norm_reg_eps <= 0:
            raise ValueError("norm_reg_eps must be positive.")
        if self.norm_reg_alpha <= 0:
            raise ValueError("norm_reg_alpha must be positive.")
        if not 0.0 <= self.text_feature_dropout < 1.0:
            raise ValueError("text_feature_dropout must be in [0, 1).")
        if self.disp_weight < 0:
            raise ValueError("disp_weight must be non-negative.")
        if self.disp_scale <= 0:
            raise ValueError("disp_scale must be positive.")
        if self.radial_shadow_slope <= 0:
            raise ValueError("radial_shadow_slope must be positive.")
        if self.radial_shadow_radial_scale <= 0:
            raise ValueError("radial_shadow_radial_scale must be positive.")
        if self.radial_shadow_radial_weight < 0:
            raise ValueError("radial_shadow_radial_weight must be non-negative.")
        if self.shadow_overlap_weight < 0:
            raise ValueError("shadow_overlap_weight must be non-negative.")
        if self.shadow_overlap_scale <= 0:
            raise ValueError("shadow_overlap_scale must be positive.")

        curv_init = float(curv_init)
        curv_min = 0.1 if curv_min is None else float(curv_min)
        curv_max = 10.0 if curv_max is None else float(curv_max)
        if curv_init <= 0 or curv_min <= 0 or curv_max <= 0 or curv_min > curv_max:
            raise ValueError("curv_init/curv_min/curv_max must be positive and ordered.")
        self.curv = nn.Parameter(torch.tensor(curv_init).log(), requires_grad=bool(learn_curv))
        self.register_buffer("_curv_min_log", torch.tensor(curv_min).log())
        self.register_buffer("_curv_max_log", torch.tensor(curv_max).log())

        # Original MERU learned tangent scaling.  No L2-fixed/BN/radius variants.
        self.visual_alpha = nn.Parameter(torch.tensor(embed_dim**-0.5).log())
        self.textual_alpha = nn.Parameter(torch.tensor(embed_dim**-0.5).log())

        if isinstance(angular_temperature_init, str) and angular_temperature_init.lower().strip() in {"", "none", "null"}:
            angular_temperature_init = None
        if angular_temperature_init is not None:
            angular_temperature_init = float(angular_temperature_init)
            if angular_temperature_init <= 0:
                raise ValueError("angular_temperature_init must be positive or None.")
            angular_logit_scale_init = 1.0 / angular_temperature_init
        ang_scale = torch.tensor(float(angular_logit_scale_init)).clamp_min(1e-9).log()
        self.angular_logit_scale = nn.Parameter(
            ang_scale,
            requires_grad=bool(
                self.entail_weight > 0
                and self.entailment_loss_type in {
                    "sigmoid_cos",
                    "contrastive_cos",
                    "typed_shadow",
                    "radial_shadow",
                    "radial_shadow_decomposed",
                }
            ),
        )
        ang_bias = torch.tensor(float(angular_logit_bias_init))
        self.angular_logit_bias = nn.Parameter(
            ang_bias,
            requires_grad=bool(
                self.entail_weight > 0
                and self.entailment_loss_type in {
                    "sigmoid_cos",
                    "typed_shadow",
                    "radial_shadow",
                    "radial_shadow_decomposed",
                }
            ),
        )

    def _clamp_hyperbolic_parameters(self) -> torch.Tensor:
        with torch.no_grad():
            self.curv.clamp_(self._curv_min_log.to(self.curv.device), self._curv_max_log.to(self.curv.device))
            self.visual_alpha.clamp_(max=0.0)
            self.textual_alpha.clamp_(max=0.0)
            self.angular_logit_scale.clamp_(max=math.log(100.0))
        return self.curv.exp()

    def _project_image_raw(self, images: torch.Tensor) -> torch.Tensor:
        return self.visual_proj(self.encode_image_backbone(images))

    def _project_text_raw(self, tokens: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        raw = self.textual_proj(self.encode_text_backbone(tokens))
        if self.training and self.text_feature_dropout > 0:
            raw = F.dropout(raw, p=self.text_feature_dropout, training=True)
        return raw

    def _image_raw_to_lorentz(self, raw: torch.Tensor, curv: torch.Tensor) -> torch.Tensor:
        with torch.autocast(self.device.type, dtype=torch.float32):
            return L.exp_map0(raw * self.visual_alpha.exp(), curv, eps=self.lorentz_eps)

    def _text_raw_to_lorentz(self, raw: torch.Tensor, curv: torch.Tensor) -> torch.Tensor:
        with torch.autocast(self.device.type, dtype=torch.float32):
            return L.exp_map0(raw * self.textual_alpha.exp(), curv, eps=self.lorentz_eps)

    def encode_image(self, images: torch.Tensor, project: bool = True) -> torch.Tensor:
        raw = self._project_image_raw(images)
        return self._image_raw_to_lorentz(raw, self.curv.exp()) if project else raw

    def encode_text(self, tokens: list[torch.Tensor] | torch.Tensor, project: bool = True) -> torch.Tensor:
        raw = self._project_text_raw(tokens)
        return self._text_raw_to_lorentz(raw, self.curv.exp()) if project else raw

    def encode_image_for_eval(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode_image(images, project=True)

    def encode_text_for_eval(self, tokens: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        # Per-prompt Lorentz embeddings; classification will average through build_text_classifier_features.
        return self.encode_text(tokens, project=True)

    def build_text_classifier_features(self, tokens: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        # Average prompts in tangent/pre-exp feature space, then map once to the manifold.
        raw = self._project_text_raw(tokens).mean(dim=0)
        return self._text_raw_to_lorentz(raw, self.curv.exp())

    def _score_type_for_eval(self) -> str:
        return self.main_score_type if self.eval_score_type == "main" else self.eval_score_type

    def compute_eval_similarity(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        curv = self.curv.exp()
        score_type = self._score_type_for_eval()
        if score_type == "lorentz_distance":
            return -pairwise_lorentz_distance(image_features, text_features, curv, eps=self.lorentz_eps)
        if score_type == "rapidity":
            return -pairwise_rapidity(image_features, text_features, curv, eps=self.lorentz_eps)
        if score_type == "root_angle":
            return pairwise_root_angle_cos(image_features, text_features, curv, eps=self.lorentz_eps)
        if score_type == "angular_cos":
            # Rows are image queries, columns are text candidates; text is always parent.
            return compute_exterior_angle_cos(text_features, image_features, curv, eps=self.lorentz_eps).T
        if score_type == "typed_shadow":
            # Rows are image queries, columns are text candidates; text is always parent.
            depth, _, _ = typed_boundary_shadow_depth(text_features, image_features, curv, eps=self.lorentz_eps)
            return depth.T
        if score_type == "radial_shadow":
            # Same text-parent -> image-leaf order; transpose only changes query axis.
            depth, _, _, _, _, _ = modality_typed_radial_shadow_depth(
                text_features,
                image_features,
                curv,
                radial_slope=self.radial_shadow_slope,
                radial_scale=self.radial_shadow_radial_scale,
                eps=self.lorentz_eps,
            )
            return depth.T
        raise ValueError(f"Unsupported eval_score_type={score_type!r}")

    def _main_scores(
        self,
        image_feats: torch.Tensor,
        text_feats: torch.Tensor,
        all_image_feats: torch.Tensor,
        all_text_feats: torch.Tensor,
        curv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.main_score_type == "lorentz_distance":
            image_values = pairwise_lorentz_distance(image_feats, all_text_feats, curv, eps=self.lorentz_eps)
            text_values = pairwise_lorentz_distance(text_feats, all_image_feats, curv, eps=self.lorentz_eps)
            return -image_values, -text_values, image_values, text_values
        if self.main_score_type == "rapidity":
            image_values = pairwise_rapidity(image_feats, all_text_feats, curv, eps=self.lorentz_eps)
            text_values = pairwise_rapidity(text_feats, all_image_feats, curv, eps=self.lorentz_eps)
            return -image_values, -text_values, image_values, text_values
        if self.main_score_type == "root_angle":
            image_scores = pairwise_root_angle_cos(image_feats, all_text_feats, curv, eps=self.lorentz_eps)
            text_scores = pairwise_root_angle_cos(text_feats, all_image_feats, curv, eps=self.lorentz_eps)
            image_values = torch.acos(image_scores.detach().clamp(-1.0 + self.lorentz_eps, 1.0 - self.lorentz_eps))
            text_values = torch.acos(text_scores.detach().clamp(-1.0 + self.lorentz_eps, 1.0 - self.lorentz_eps))
            return image_scores, text_scores, image_values, text_values
        raise AssertionError("main_score_type already validated")

    def _adaptive_norm_penalty(self, norm: torch.Tensor, mode: str) -> torch.Tensor:
        """Self-decaying penalty that discourages small norms without a hard margin."""
        if mode.startswith("inverse_square_"):
            return norm.reciprocal().square()
        if mode.startswith("inverse_"):
            return norm.reciprocal()
        if mode.startswith("softplus_neg_"):
            return F.softplus(-self.norm_reg_alpha * norm)
        raise AssertionError("adaptive norm mode already validated")

    def _compute_norm_regularizer(
        self,
        text_feats: torch.Tensor,
        image_feats: torch.Tensor,
        curv: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute optional norm penalty and always report cheap spatial-norm stats.

        Spatial norms are O(BD) and are useful diagnostics even when regularization
        is disabled.  Radius/acosh quantities remain strictly on-demand.
        """
        zero = text_feats.new_zeros(())
        eps = self.lorentz_eps
        text_space = torch.norm(text_feats, dim=-1)
        image_space = torch.norm(image_feats, dim=-1)
        logging = {
            "norm/loss": zero.detach(),
            "norm/text_space_norm_mean": text_space.detach().mean(),
            "norm/image_space_norm_mean": image_space.detach().mean(),
            "norm/text_space_norm_min": text_space.detach().min(),
            "norm/image_space_norm_min": image_space.detach().min(),
            "norm/text_space_norm_max": text_space.detach().max(),
            "norm/image_space_norm_max": image_space.detach().max(),
        }

        if self.norm_weight <= 0 or self.norm_reg_type == "none":
            return zero, logging

        text_radius = image_radius = None
        if self.norm_reg_type == "argent_space":
            t = text_space.clamp_min(eps)
            v = image_space.clamp_min(eps)
            loss_norm = 0.5 * ((t * t - 0.5 * torch.log(t)).mean() + (v * v - 0.5 * torch.log(v)).mean())
            # loss_norm = 0.5 * ((t * t -torch.log(t)).mean() + (v * v - torch.log(v)).mean())
        elif self.norm_reg_type in {"inverse_space", "inverse_square_space", "softplus_neg_space"}:
            t = text_space.clamp_min(self.norm_reg_eps)
            v = image_space.clamp_min(self.norm_reg_eps)
            loss_norm = 0.5 * (
                self._adaptive_norm_penalty(t, self.norm_reg_type).mean()
                + self._adaptive_norm_penalty(v, self.norm_reg_type).mean()
            )
        elif self.norm_reg_type == "argent_radius":
            text_radius = radius_from_origin(text_feats, curv, eps=eps)
            image_radius = radius_from_origin(image_feats, curv, eps=eps)
            t = (torch.sqrt(curv) * text_radius).clamp_min(eps)
            v = (torch.sqrt(curv) * image_radius).clamp_min(eps)
            loss_norm = 0.5 * ((t * t - torch.log(t)).mean() + (v * v - torch.log(v)).mean())
        elif self.norm_reg_type in {"inverse_radius", "inverse_square_radius", "softplus_neg_radius"}:
            text_radius = radius_from_origin(text_feats, curv, eps=eps)
            image_radius = radius_from_origin(image_feats, curv, eps=eps)
            t = (torch.sqrt(curv) * text_radius).clamp_min(self.norm_reg_eps)
            v = (torch.sqrt(curv) * image_radius).clamp_min(self.norm_reg_eps)
            loss_norm = 0.5 * (
                self._adaptive_norm_penalty(t, self.norm_reg_type).mean()
                + self._adaptive_norm_penalty(v, self.norm_reg_type).mean()
            )
        else:
            raise AssertionError("norm_reg_type already validated")

        logging["norm/loss"] = loss_norm.detach()
        logging["norm/reg_alpha"] = text_feats.new_tensor(self.norm_reg_alpha)
        if text_radius is not None and image_radius is not None:
            logging.update({
                "norm/text_radius_mean": text_radius.detach().mean(),
                "norm/image_radius_mean": image_radius.detach().mean(),
                "norm/text_radius_std": text_radius.detach().std(unbiased=False),
                "norm/image_radius_std": image_radius.detach().std(unbiased=False),
                "norm/text_radius_min": text_radius.detach().min(),
                "norm/image_radius_min": image_radius.detach().min(),
                "norm/text_radius_max": text_radius.detach().max(),
                "norm/image_radius_max": image_radius.detach().max(),
            })
        return loss_norm, logging

    def _positive_geometry_diagnostics(
        self,
        text_feats: torch.Tensor,
        image_feats: torch.Tensor,
        curv: torch.Tensor,
        pos_root_cos: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        with torch.autocast(self.device.type, dtype=torch.float32):
            if pos_root_cos is None:
                pos_root_cos = F.cosine_similarity(text_feats, image_feats, dim=-1, eps=self.lorentz_eps)
            pos_root_cos = pos_root_cos.clamp(-1.0 + self.lorentz_eps, 1.0 - self.lorentz_eps)
            pos_root_angle = torch.acos(pos_root_cos.detach())
            out = {
                "pos_root_angle_cos_mean": pos_root_cos.detach().mean(),
                "pos_root_angle_mean": pos_root_angle.mean(),
            }
            # Exterior/cone diagnostics are useful only when the entailment branch
            # is active; avoiding them in the pure root-angle route saves an extra
            # acosh/acos-heavy pass every iteration.
            if self.entail_weight > 0 and self.entailment_loss_type in {"positive_cos", "sigmoid_cos", "contrastive_cos"}:
                pos_alpha = torch.acos(
                    compute_paired_exterior_angle_cos(text_feats, image_feats, curv, eps=self.lorentz_eps).detach()
                )
                pos_aperture = L.half_aperture(text_feats, curv, eps=self.lorentz_eps)
                ext_minus = pos_alpha - pos_aperture
                out.update({
                    "pos_aperture_mean": pos_aperture.detach().mean(),
                    "pos_inside_cone_ratio": (ext_minus.detach() <= 0).float().mean(),
                    "pos_ext_minus_aper_mean": ext_minus.detach().mean(),
                    "pos_old_cone_alpha_mean": pos_alpha.detach().mean(),
                    "pos_old_cone_aperture_mean": pos_aperture.detach().mean(),
                    "pos_old_cone_inside_ratio": (ext_minus.detach() <= 0).float().mean(),
                    "pos_root_angle_lt_aperture_ratio": (pos_root_angle <= pos_aperture.detach()).float().mean(),
                })
            return out

    def _angular_topk_enabled(self, num_cols: int) -> bool:
        return (
            self.angular_negative_mining == "topk"
            and self.angular_topk > 0
            and self.angular_topk < num_cols - 1
        )

    def _angular_sigmoid_loss_from_logits(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Sigmoid angular loss with optional hard-negative mining and reduction."""
        num_rows, num_cols = logits.shape
        row_ids = torch.arange(num_rows, device=logits.device)
        pos_logits = logits[row_ids, targets]
        pos_loss = F.softplus(-pos_logits)

        pos_mask = torch.zeros_like(logits, dtype=torch.bool)
        pos_mask[row_ids, targets] = True

        if self._angular_topk_enabled(num_cols):
            neg_source = logits.masked_fill(pos_mask, float("-inf"))
            neg_logits = torch.topk(neg_source, k=self.angular_topk, dim=1).values
        else:
            neg_logits = logits.masked_select(~pos_mask).view(num_rows, num_cols - 1)

        neg_loss = F.softplus(neg_logits)
        if self.angular_reduction == "balanced":
            loss_ang = 0.5 * (pos_loss.mean() + neg_loss.mean())
        else:
            loss_sum = pos_loss.sum() + neg_loss.sum()
            if self.sigmoid_normalization == "pairs":
                denom = num_rows * (neg_logits.shape[1] + 1)
            else:
                denom = num_rows
            loss_ang = loss_sum / denom

        logging = {
            "ang_pos_pair_loss_mean": pos_loss.detach().mean(),
            "ang_neg_pair_loss_mean": neg_loss.detach().mean(),
            "ang_neg_used_per_sample": logits.new_tensor(float(neg_logits.shape[1])),
        }
        return loss_ang, logging

    def _angular_contrastive_loss_from_logits(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """CE angular loss; optional top-k restricts each row to positive + hard negatives."""
        num_rows, num_cols = logits.shape
        row_ids = torch.arange(num_rows, device=logits.device)
        if self._angular_topk_enabled(num_cols):
            pos_logits = logits[row_ids, targets].unsqueeze(1)
            pos_mask = torch.zeros_like(logits, dtype=torch.bool)
            pos_mask[row_ids, targets] = True
            neg_source = logits.masked_fill(pos_mask, float("-inf"))
            neg_logits = torch.topk(neg_source, k=self.angular_topk, dim=1).values
            sampled_logits = torch.cat([pos_logits, neg_logits], dim=1)
            sampled_targets = torch.zeros(num_rows, dtype=torch.long, device=logits.device)
            loss_ang = F.cross_entropy(sampled_logits, sampled_targets)
            used = neg_logits.shape[1]
        else:
            loss_ang = F.cross_entropy(logits, targets)
            used = num_cols - 1
        return loss_ang, {"ang_neg_used_per_sample": logits.new_tensor(float(used))}

    def _compute_typed_shadow_entailment(
        self,
        text_feats: torch.Tensor,
        all_image_feats: torch.Tensor,
        targets: torch.Tensor,
        curv: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        depth_pair, theta_pair, aperture = typed_boundary_shadow_depth(
            text_feats, all_image_feats, curv, eps=self.lorentz_eps
        )
        row_ids = torch.arange(text_feats.shape[0], device=text_feats.device)
        pos_depth = depth_pair[row_ids, targets]
        pos_theta = theta_pair[row_ids, targets]

        scale = self.angular_logit_scale.exp()
        with torch.no_grad():
            self.angular_logit_scale.clamp_(max=math.log(100.0))
        logits = scale * depth_pair + self.angular_logit_bias

        if self.angular_reduction == "pair_mean" and not self._angular_topk_enabled(depth_pair.shape[1]):
            loss_ang = sigmoid_loss_from_logits(
                logits,
                targets=targets,
                normalization=self.sigmoid_normalization,
                negatives_per_sample=0,
            )
            loss_logging = {
                "ang_neg_used_per_sample": logits.new_tensor(float(logits.shape[1] - 1)),
            }
        else:
            loss_ang, loss_logging = self._angular_sigmoid_loss_from_logits(logits, targets)

        mask = torch.ones_like(depth_pair, dtype=torch.bool)
        mask[row_ids, targets] = False
        neg_depth = depth_pair[mask]
        neg_theta = theta_pair[mask]
        if neg_depth.numel() > 0:
            neg_depth_mean = neg_depth.detach().mean()
            neg_theta_mean = neg_theta.detach().mean()
            neg_cos_mean = torch.cos(neg_theta.detach()).mean()
        else:
            neg_depth_mean = text_feats.new_zeros(())
            neg_theta_mean = text_feats.new_zeros(())
            neg_cos_mean = text_feats.new_zeros(())

        logging = {
            "loss_ang": loss_ang.detach(),
            "shadow_pos_depth_mean": pos_depth.detach().mean(),
            "shadow_neg_depth_mean": neg_depth_mean,
            "shadow_text_aperture_mean": aperture.detach().mean(),
            "shadow_text_aperture_min": aperture.detach().min(),
            "shadow_text_aperture_max": aperture.detach().max(),
            "shadow_pos_theta_mean": pos_theta.detach().mean(),
            "shadow_neg_theta_mean": neg_theta_mean,
            "shadow_pos_inside_ratio": (pos_depth.detach() >= 0).float().mean(),
            "ang_pos_depth_mean": pos_depth.detach().mean(),
            "ang_neg_depth_mean": neg_depth_mean,
            "ang_pos_cos_mean": torch.cos(pos_theta.detach()).mean(),
            "ang_neg_cos_mean": neg_cos_mean,
            "ang_pos_alpha_mean": pos_theta.detach().mean(),
            "ang_neg_alpha_mean": neg_theta_mean,
            "ang_logit_scale": scale.detach(),
            "ang_temperature": (1.0 / scale).detach(),
            "ang_logit_bias": self.angular_logit_bias.detach(),
        }
        logging.update(loss_logging)
        return loss_ang, logging

    def _compute_radial_shadow_entailment(
        self,
        text_feats: torch.Tensor,
        all_image_feats: torch.Tensor,
        targets: torch.Tensor,
        curv: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        depth_pair, theta_pair, allowed_angle, radial_gap, rho_text, rho_image = (
            modality_typed_radial_shadow_depth(
                text_feats,
                all_image_feats,
                curv,
                radial_slope=self.radial_shadow_slope,
                radial_scale=self.radial_shadow_radial_scale,
                eps=self.lorentz_eps,
            )
        )
        row_ids = torch.arange(text_feats.shape[0], device=text_feats.device)
        pos_depth = depth_pair[row_ids, targets]
        pos_theta = theta_pair[row_ids, targets]
        pos_allowed = allowed_angle[row_ids, targets]
        pos_gap = radial_gap[row_ids, targets]

        scale = self.angular_logit_scale.exp()
        with torch.no_grad():
            self.angular_logit_scale.clamp_(max=math.log(100.0))
        logits = scale * depth_pair + self.angular_logit_bias

        if self.angular_reduction == "pair_mean" and not self._angular_topk_enabled(depth_pair.shape[1]):
            loss_ang = sigmoid_loss_from_logits(
                logits,
                targets=targets,
                normalization=self.sigmoid_normalization,
                negatives_per_sample=0,
            )
            loss_logging = {
                "ang_neg_used_per_sample": logits.new_tensor(float(logits.shape[1] - 1)),
            }
        else:
            loss_ang, loss_logging = self._angular_sigmoid_loss_from_logits(logits, targets)

        mask = torch.ones_like(depth_pair, dtype=torch.bool)
        mask[row_ids, targets] = False
        neg_depth = depth_pair[mask]
        neg_theta = theta_pair[mask]
        neg_gap = radial_gap[mask]
        neg_allowed = allowed_angle[mask]
        if neg_depth.numel() > 0:
            neg_depth_mean = neg_depth.detach().mean()
            neg_theta_mean = neg_theta.detach().mean()
            neg_gap_mean = neg_gap.detach().mean()
            neg_allowed_mean = neg_allowed.detach().mean()
            neg_cos_mean = torch.cos(neg_theta.detach()).mean()
        else:
            zero = text_feats.new_zeros(())
            neg_depth_mean = neg_theta_mean = neg_gap_mean = neg_allowed_mean = neg_cos_mean = zero

        pos_radial_ok = pos_gap.detach() >= 0
        pos_angular_ok = pos_theta.detach() <= pos_allowed.detach()
        logging = {
            "loss_ang": loss_ang.detach(),
            "radial_shadow_pos_depth_mean": pos_depth.detach().mean(),
            "radial_shadow_neg_depth_mean": neg_depth_mean,
            "radial_shadow_pos_radial_gap_mean": pos_gap.detach().mean(),
            "radial_shadow_neg_radial_gap_mean": neg_gap_mean,
            "radial_shadow_pos_allowed_angle_mean": pos_allowed.detach().mean(),
            "radial_shadow_neg_allowed_angle_mean": neg_allowed_mean,
            "radial_shadow_pos_theta_mean": pos_theta.detach().mean(),
            "radial_shadow_neg_theta_mean": neg_theta_mean,
            "radial_shadow_pos_radial_order_ratio": pos_radial_ok.float().mean(),
            "radial_shadow_pos_angular_inside_ratio": pos_angular_ok.float().mean(),
            "radial_shadow_pos_order_ratio": (pos_radial_ok & pos_angular_ok).float().mean(),
            "radial_shadow_text_rho_mean": rho_text.detach().mean(),
            "radial_shadow_image_rho_mean": rho_image.detach().mean(),
            "radial_shadow_slope": text_feats.new_tensor(self.radial_shadow_slope),
            "radial_shadow_radial_scale": text_feats.new_tensor(self.radial_shadow_radial_scale),
            "ang_pos_depth_mean": pos_depth.detach().mean(),
            "ang_neg_depth_mean": neg_depth_mean,
            "ang_pos_cos_mean": torch.cos(pos_theta.detach()).mean(),
            "ang_neg_cos_mean": neg_cos_mean,
            "ang_pos_alpha_mean": pos_theta.detach().mean(),
            "ang_neg_alpha_mean": neg_theta_mean,
            "ang_logit_scale": scale.detach(),
            "ang_temperature": (1.0 / scale).detach(),
            "ang_logit_bias": self.angular_logit_bias.detach(),
        }
        logging.update(loss_logging)
        return loss_ang, logging

    def _compute_decomposed_radial_shadow_entailment(
        self,
        text_feats: torch.Tensor,
        all_image_feats: torch.Tensor,
        targets: torch.Tensor,
        curv: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Decomposed surrogate for the same Modality-Typed Radial Shadow order.

        Geometry is unchanged: a valid text->image pair must satisfy both the
        radial type order and angular shadow membership.  The legacy branch
        compresses both certificates with ``min`` before sigmoid training.  This
        branch instead optimizes:

          * pairwise sigmoid shadow membership for positive/negative matching;
          * positive-only soft radial type order for rho_image > rho_text.

        The aperture is detached only in the shadow path.  Hence angular
        membership cannot widen its cone by manipulating norms, while the radial
        component remains responsible for learning modality depth.
        """
        joint_depth, theta_pair, allowed_angle, radial_gap, rho_text, rho_image = (
            modality_typed_radial_shadow_depth(
                text_feats,
                all_image_feats,
                curv,
                radial_slope=self.radial_shadow_slope,
                radial_scale=self.radial_shadow_radial_scale,
                eps=self.lorentz_eps,
            )
        )
        row_ids = torch.arange(text_feats.shape[0], device=text_feats.device)

        # Instance-level matching is carried by angular shadow membership.
        # Stop-gradient on aperture prevents the shadow objective from using norm
        # inflation as a shortcut.  The decision boundary remains exactly
        # theta <= allowed_angle.
        shadow_margin = (
            (allowed_angle.detach() - theta_pair) / math.pi
        ).clamp(-1.0, 1.0)
        pos_shadow_margin = shadow_margin[row_ids, targets]

        scale = self.angular_logit_scale.exp()
        with torch.no_grad():
            self.angular_logit_scale.clamp_(max=math.log(100.0))
        shadow_logits = scale * shadow_margin + self.angular_logit_bias

        if self.angular_reduction == "pair_mean" and not self._angular_topk_enabled(shadow_margin.shape[1]):
            loss_shadow = sigmoid_loss_from_logits(
                shadow_logits,
                targets=targets,
                normalization=self.sigmoid_normalization,
                negatives_per_sample=0,
            )
            loss_logging = {
                "ang_neg_used_per_sample": shadow_logits.new_tensor(float(shadow_logits.shape[1] - 1)),
            }
        else:
            loss_shadow, loss_logging = self._angular_sigmoid_loss_from_logits(
                shadow_logits, targets
            )

        # Radial order is a modality-type relation, not a positive-vs-negative
        # image discriminator.  Optimize it only on aligned pairs.
        pos_gap = radial_gap[row_ids, targets]
        if self.radial_shadow_radial_weight > 0:
            loss_radial = F.softplus(
                -pos_gap / self.radial_shadow_radial_scale
            ).mean()
        else:
            loss_radial = text_feats.new_zeros(())
        loss_ang = loss_shadow + self.radial_shadow_radial_weight * loss_radial

        mask = torch.ones_like(shadow_margin, dtype=torch.bool)
        mask[row_ids, targets] = False
        neg_shadow_margin = shadow_margin[mask]
        neg_joint_depth = joint_depth[mask]
        neg_theta = theta_pair[mask]
        neg_gap = radial_gap[mask]
        neg_allowed = allowed_angle[mask]
        if neg_shadow_margin.numel() > 0:
            neg_shadow_mean = neg_shadow_margin.detach().mean()
            neg_joint_mean = neg_joint_depth.detach().mean()
            neg_theta_mean = neg_theta.detach().mean()
            neg_gap_mean = neg_gap.detach().mean()
            neg_allowed_mean = neg_allowed.detach().mean()
            neg_cos_mean = torch.cos(neg_theta.detach()).mean()
        else:
            zero = text_feats.new_zeros(())
            neg_shadow_mean = neg_joint_mean = neg_theta_mean = zero
            neg_gap_mean = neg_allowed_mean = neg_cos_mean = zero

        pos_theta = theta_pair[row_ids, targets]
        pos_allowed = allowed_angle[row_ids, targets]
        pos_joint_depth = joint_depth[row_ids, targets]
        pos_radial_ok = pos_gap.detach() >= 0
        pos_angular_ok = pos_theta.detach() <= pos_allowed.detach()
        logging = {
            "loss_ang": loss_ang.detach(),
            "loss_shadow_membership": loss_shadow.detach(),
            "loss_radial_order": loss_radial.detach(),
            "radial_shadow/radial_weight": text_feats.new_tensor(self.radial_shadow_radial_weight),
            "radial_shadow_pos_depth_mean": pos_joint_depth.detach().mean(),
            "radial_shadow_neg_depth_mean": neg_joint_mean,
            "radial_shadow_pos_shadow_margin_mean": pos_shadow_margin.detach().mean(),
            "radial_shadow_neg_shadow_margin_mean": neg_shadow_mean,
            "radial_shadow_pos_radial_gap_mean": pos_gap.detach().mean(),
            "radial_shadow_neg_radial_gap_mean": neg_gap_mean,
            "radial_shadow_pos_allowed_angle_mean": pos_allowed.detach().mean(),
            "radial_shadow_neg_allowed_angle_mean": neg_allowed_mean,
            "radial_shadow_pos_theta_mean": pos_theta.detach().mean(),
            "radial_shadow_neg_theta_mean": neg_theta_mean,
            "radial_shadow_pos_radial_order_ratio": pos_radial_ok.float().mean(),
            "radial_shadow_pos_angular_inside_ratio": pos_angular_ok.float().mean(),
            "radial_shadow_pos_order_ratio": (pos_radial_ok & pos_angular_ok).float().mean(),
            "radial_shadow_text_rho_mean": rho_text.detach().mean(),
            "radial_shadow_image_rho_mean": rho_image.detach().mean(),
            "radial_shadow_slope": text_feats.new_tensor(self.radial_shadow_slope),
            "radial_shadow_radial_scale": text_feats.new_tensor(self.radial_shadow_radial_scale),
            "ang_pos_depth_mean": pos_shadow_margin.detach().mean(),
            "ang_neg_depth_mean": neg_shadow_mean,
            "ang_pos_cos_mean": torch.cos(pos_theta.detach()).mean(),
            "ang_neg_cos_mean": neg_cos_mean,
            "ang_pos_alpha_mean": pos_theta.detach().mean(),
            "ang_neg_alpha_mean": neg_theta_mean,
            "ang_logit_scale": scale.detach(),
            "ang_temperature": (1.0 / scale).detach(),
            "ang_logit_bias": self.angular_logit_bias.detach(),
        }
        logging.update(loss_logging)
        return loss_ang, logging

    def _compute_shadow_overlap_loss(
        self,
        text_feats: torch.Tensor,
        image_feats: torch.Tensor,
        curv: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Penalize incorrect directed containment between local text shadows.

        Ordinary semantic overlap is allowed.  For text shadows B_i and B_j,
        only the directed condition B_j subset B_i is considered a violation:

            theta_ij <= max(psi_i - psi_j, 0).

        The implementation works entirely in cosine space and masks the diagonal
        before reduction.  It therefore has no acos(1) gradient singularity.
        Apertures are detached, so this auxiliary branch separates text
        directions without inflating or contracting text/image norms.
        """
        zero = text_feats.new_zeros(())
        if self.shadow_overlap_weight <= 0 or text_feats.shape[0] <= 1:
            return zero, {"loss_shadow_overlap": zero.detach()}

        with torch.no_grad():
            rho_text = normalized_radius_from_origin(
                text_feats, curv, eps=self.lorentz_eps
            )
            rho_image = normalized_radius_from_origin(
                image_feats, curv, eps=self.lorentz_eps
            )
            frontier = rho_image.mean()
            aperture = (
                self.radial_shadow_slope
                * (frontier - rho_text).clamp_min(0.0)
            ).clamp_max(math.pi)

        text_cos = pairwise_root_angle_cos(
            text_feats, text_feats, curv=curv, eps=self.lorentz_eps
        )
        # Directed cap containment B_j subset B_i can happen only when psi_i >=
        # psi_j and theta_ij <= psi_i-psi_j.  Since cosine is decreasing on
        # [0,pi], positive score means a containment violation.
        directed_aperture_gap = (
            aperture[:, None] - aperture[None, :]
        ).clamp(min=0.0, max=math.pi)
        containment_threshold_cos = torch.cos(directed_aperture_gap)
        containment_score = text_cos - containment_threshold_cos

        mask = ~torch.eye(
            text_feats.shape[0], dtype=torch.bool, device=text_feats.device
        )
        offdiag_score = containment_score[mask]
        offdiag_violation = F.relu(offdiag_score)
        offdiag_cos = text_cos[mask]
        offdiag_gap = directed_aperture_gap[mask]

        # Exactly zero outside the violation set, unlike a plain softplus over
        # every text pair.  Subtract the constant baseline so zero violation has
        # zero loss while preserving a smooth positive penalty inside.
        baseline = F.softplus(
            offdiag_violation.new_tensor(self.shadow_overlap_bias)
        )
        loss_overlap = (
            F.softplus(
                self.shadow_overlap_scale * offdiag_violation
                + self.shadow_overlap_bias
            )
            - baseline
        ).mean()

        violation_mask = offdiag_score.detach() > 0
        logging = {
            "loss_shadow_overlap": loss_overlap.detach(),
            # Backward-compatible keys now refer to directed containment.
            "shadow_overlap/margin_mean": offdiag_score.detach().mean(),
            "shadow_overlap/ratio": violation_mask.float().mean(),
            "shadow_overlap/containment_score_mean": offdiag_score.detach().mean(),
            "shadow_overlap/violation_mean": offdiag_violation.detach().mean(),
            "shadow_overlap/violation_ratio": violation_mask.float().mean(),
            "shadow_overlap/text_cos_mean": offdiag_cos.detach().mean(),
            "shadow_overlap/directed_aperture_gap_mean": offdiag_gap.detach().mean(),
            "shadow_overlap/aperture_mean": aperture.detach().mean(),
            "shadow_overlap/frontier_rho": frontier.detach(),
            "shadow_overlap/scale": text_feats.new_tensor(self.shadow_overlap_scale),
            "shadow_overlap/bias": text_feats.new_tensor(self.shadow_overlap_bias),
        }
        return loss_overlap, logging

    def _compute_dispersive_loss(
        self,
        all_image_feats: torch.Tensor,
        all_text_feats: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        zero = all_image_feats.new_zeros(())
        if self.disp_weight <= 0:
            return zero, {"loss_disp": zero.detach()}

        def _single_modality_loss(feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            num = feats.shape[0]
            if num <= 1:
                return feats.new_zeros(()), feats.new_zeros(())
            cos_pair = pairwise_root_angle_cos(feats, feats, eps=self.lorentz_eps)
            mask = ~torch.eye(num, dtype=torch.bool, device=feats.device)
            neg_cos = cos_pair[mask]
            loss = F.softplus(self.disp_scale * neg_cos + self.disp_bias).mean()
            return loss, neg_cos.detach().mean()

        image_loss, image_cos_mean = _single_modality_loss(all_image_feats)
        text_loss, text_cos_mean = _single_modality_loss(all_text_feats)
        loss_disp = image_loss + text_loss
        logging = {
            "loss_disp": loss_disp.detach(),
            "disp/image_loss": image_loss.detach(),
            "disp/text_loss": text_loss.detach(),
            "disp/image_cos_mean": image_cos_mean,
            "disp/text_cos_mean": text_cos_mean,
            "disp/scale": all_image_feats.new_tensor(float(self.disp_scale)),
            "disp/bias": all_image_feats.new_tensor(float(self.disp_bias)),
        }
        return loss_disp, logging

    def _compute_angular_entailment(
        self,
        text_feats: torch.Tensor,
        image_feats: torch.Tensor,
        all_image_feats: torch.Tensor,
        targets: torch.Tensor,
        curv: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.entail_weight <= 0 or self.entailment_loss_type == "none":
            zero = text_feats.new_zeros(())
            return zero, {"loss_ang": zero.detach()}

        row_ids = torch.arange(text_feats.shape[0], device=text_feats.device)

        if self.entailment_loss_type == "positive_cos":
            pos_cos = compute_paired_exterior_angle_cos(text_feats, image_feats, curv, eps=self.lorentz_eps)
            pos_alpha = torch.acos(pos_cos.detach())
            loss_ang = (1.0 - pos_cos).mean()
            logging = {
                "loss_ang": loss_ang.detach(),
                "ang_pos_cos_mean": pos_cos.detach().mean(),
                "ang_pos_alpha_mean": pos_alpha.mean(),
            }
            return loss_ang, logging

        if self.entailment_loss_type == "typed_shadow":
            return self._compute_typed_shadow_entailment(
                text_feats=text_feats,
                all_image_feats=all_image_feats,
                targets=targets,
                curv=curv,
            )

        if self.entailment_loss_type == "radial_shadow":
            return self._compute_radial_shadow_entailment(
                text_feats=text_feats,
                all_image_feats=all_image_feats,
                targets=targets,
                curv=curv,
            )

        if self.entailment_loss_type == "radial_shadow_decomposed":
            return self._compute_decomposed_radial_shadow_entailment(
                text_feats=text_feats,
                all_image_feats=all_image_feats,
                targets=targets,
                curv=curv,
            )

        # Pairwise exterior-angle scores for T2I entailment. Text rows are parents,
        # all-image columns are children. Reuse the same cos(alpha) matrix for both
        # CLIP-like softmax contrastive and SigLIP-like pairwise sigmoid variants.
        cos_pair = compute_exterior_angle_cos(text_feats, all_image_feats, curv, eps=self.lorentz_eps)
        pos_cos = cos_pair[row_ids, targets]
        pos_alpha = torch.acos(pos_cos.detach())
        scale = self.angular_logit_scale.exp()
        with torch.no_grad():
            self.angular_logit_scale.clamp_(max=math.log(100.0))

        if self.entailment_loss_type == "contrastive_cos":
            logits = scale * cos_pair
            loss_ang, loss_logging = self._angular_contrastive_loss_from_logits(logits, targets)
        else:
            logits = scale * cos_pair + self.angular_logit_bias
            if self.angular_reduction == "pair_mean" and not self._angular_topk_enabled(cos_pair.shape[1]):
                loss_ang = sigmoid_loss_from_logits(
                    logits,
                    targets=targets,
                    normalization=self.sigmoid_normalization,
                    negatives_per_sample=0,
                )
                loss_logging = {
                    "ang_neg_used_per_sample": logits.new_tensor(float(logits.shape[1] - 1)),
                }
            else:
                loss_ang, loss_logging = self._angular_sigmoid_loss_from_logits(logits, targets)

        mask = torch.ones_like(cos_pair, dtype=torch.bool)
        mask[row_ids, targets] = False
        neg_cos = cos_pair[mask]
        if neg_cos.numel() > 0:
            neg_cos_mean = neg_cos.detach().mean()
            neg_alpha_mean = torch.acos(neg_cos.detach()).mean()
        else:
            neg_cos_mean = text_feats.new_zeros(())
            neg_alpha_mean = text_feats.new_zeros(())
        logging = {
            "loss_ang": loss_ang.detach(),
            "ang_pos_cos_mean": pos_cos.detach().mean(),
            "ang_neg_cos_mean": neg_cos_mean,
            "ang_pos_alpha_mean": pos_alpha.mean(),
            "ang_neg_alpha_mean": neg_alpha_mean,
            "ang_logit_scale": scale.detach(),
            "ang_temperature": (1.0 / scale).detach(),
            "ang_logit_bias": self.angular_logit_bias.detach() if self.entailment_loss_type == "sigmoid_cos" else text_feats.new_zeros(()),
        }
        logging.update(loss_logging)
        return loss_ang, logging

    def forward(
        self,
        images: torch.Tensor,
        tokens: list[torch.Tensor] | torch.Tensor,
        iteration: int | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        curv = self._clamp_hyperbolic_parameters()
        main_scale = self._clamp_logit_scale()
        batch_size = images.shape[0]

        image_feats = self.encode_image(images, project=True)
        text_feats = self.encode_text(tokens, project=True)
        all_image_feats = torch.cat(dist.gather_across_processes(image_feats), dim=0)
        all_text_feats = torch.cat(dist.gather_across_processes(text_feats), dim=0)

        with torch.autocast(self.device.type, dtype=torch.float32):
            image_scores, text_scores, image_values, text_values = self._main_scores(
                image_feats, text_feats, all_image_feats, all_text_feats, curv
            )
            targets = shifted_targets(batch_size, self._rank, image_scores.device)
            loss_main = image_text_loss(
                self.loss_type,
                image_scores,
                text_scores,
                targets=targets,
                scale=main_scale,
                bias=self.logit_bias if self.loss_type == "sigmoid" else None,
                sigmoid_normalization=self.sigmoid_normalization,
                sigmoid_negatives_per_sample=self.sigmoid_negatives_per_sample,
            )
            loss_ang, ang_logging = self._compute_angular_entailment(
                text_feats=text_feats,
                image_feats=image_feats,
                all_image_feats=all_image_feats,
                targets=targets,
                curv=curv,
            )
            loss_norm, norm_logging = self._compute_norm_regularizer(text_feats, image_feats, curv)
            loss_disp, disp_logging = self._compute_dispersive_loss(all_image_feats, all_text_feats)
            loss_shadow_overlap, shadow_overlap_logging = self._compute_shadow_overlap_loss(
                text_feats, image_feats, curv
            )
            diag_pos_root_cos = None
            if self.main_score_type == "root_angle":
                diag_rows = torch.arange(batch_size, device=image_scores.device)
                diag_pos_root_cos = image_scores.detach()[diag_rows, targets]
            geometry_logging = self._positive_geometry_diagnostics(
                text_feats, image_feats, curv, pos_root_cos=diag_pos_root_cos
            )

        total_loss = loss_main
        if self.entail_weight > 0 and self.entailment_loss_type != "none":
            total_loss = total_loss + self.entail_weight * loss_ang
        if self.norm_weight > 0 and self.norm_reg_type != "none":
            total_loss = total_loss + self.norm_weight * loss_norm
        if self.disp_weight > 0:
            total_loss = total_loss + self.disp_weight * loss_disp
        if self.shadow_overlap_weight > 0:
            total_loss = total_loss + self.shadow_overlap_weight * loss_shadow_overlap

        row_ids = torch.arange(batch_size, device=image_scores.device)
        pos_score = 0.5 * (
            image_scores.detach()[row_ids, targets].mean()
            + text_scores.detach()[row_ids, targets].mean()
        )
        all_score = 0.5 * (image_scores.detach().mean() + text_scores.detach().mean())
        pos_value = 0.5 * (
            image_values.detach()[row_ids, targets].mean()
            + text_values.detach()[row_ids, targets].mean()
        )
        all_value = 0.5 * (image_values.detach().mean() + text_values.detach().mean())

        logging: dict[str, torch.Tensor] = {
            "loss_main": loss_main.detach(),
            "alignment_loss": loss_main.detach(),
            f"{self.loss_type}_loss": loss_main.detach(),
            "main_score_pos_mean": pos_score,
            "main_score_neg_mean": all_score,
            "main_value_pos_mean": pos_value,
            "main_value_all_mean": all_value,
            "main_logit_scale": main_scale.detach(),
            "main_logit_bias": self.logit_bias.detach() if self.loss_type == "sigmoid" else image_scores.new_zeros(()),
            "logit_scale": main_scale.detach(),
            "curv": curv.detach(),
            "visual_alpha": self.visual_alpha.exp().detach(),
            "textual_alpha": self.textual_alpha.exp().detach(),
        }
        if self.loss_type == "sigmoid":
            logging["logit_bias"] = self.logit_bias.detach()
        logging.update(ang_logging)
        if "ang_logit_scale" not in logging:
            ang_scale = self.angular_logit_scale.exp().detach()
            logging["ang_logit_scale"] = ang_scale
            logging["ang_temperature"] = 1.0 / ang_scale
            logging["ang_logit_bias"] = self.angular_logit_bias.detach()
        logging.update(norm_logging)
        logging.update(disp_logging)
        logging.update(shadow_overlap_logging)
        logging.update(geometry_logging)
        return {"loss": total_loss, "logging": logging}


class HyCoCLIP(MERU):
    """HyCoCLIP compatibility path with image/text and localized box image/text."""

    def __init__(
        self,
        visual: nn.Module,
        textual: TransformerTextEncoder,
        embed_dim: int,
        curv_init: float = 1.0,
        learn_curv: bool = True,
        curv_min: float | None = 0.1,
        curv_max: float | None = 10.0,
        entail_weight: float = 0.0,
        use_boxes: bool = True,
        lorentz_eps: float = 1e-8,
        pixel_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        pixel_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        loss_type: str = "contrastive",
        sigmoid_logit_bias_init: float = 10.0,
        sigmoid_normalization: str = "batch",
        sigmoid_negatives_per_sample: int = 0,
        main_score_type: str = "lorentz_distance",
        entailment_loss_type: str = "none",
        norm_reg_type: str = "none",
        norm_weight: float = 0.0,
        **kwargs: object,
    ):
        del entailment_loss_type, norm_reg_type, norm_weight
        super().__init__(
            visual=visual,
            textual=textual,
            embed_dim=embed_dim,
            curv_init=curv_init,
            learn_curv=learn_curv,
            curv_min=curv_min,
            curv_max=curv_max,
            entail_weight=entail_weight,
            use_boxes=use_boxes,
            lorentz_eps=lorentz_eps,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            loss_type=loss_type,
            sigmoid_logit_bias_init=sigmoid_logit_bias_init,
            sigmoid_normalization=sigmoid_normalization,
            sigmoid_negatives_per_sample=sigmoid_negatives_per_sample,
            main_score_type=main_score_type,
            entailment_loss_type="none",
            norm_reg_type="none",
            **kwargs,
        )
        assert use_boxes, "HyCoCLIP requires box images/texts."

    def forward(
        self,
        images: torch.Tensor,
        box_images: torch.Tensor,
        tokens: list[torch.Tensor] | torch.Tensor,
        box_tokens: list[torch.Tensor] | torch.Tensor,
        iteration: int | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        curv = self._clamp_hyperbolic_parameters()
        scale = self._clamp_logit_scale()
        image_feats = self.encode_image(images, project=True)
        text_feats = self.encode_text(tokens, project=True)
        box_image_feats = self.encode_image(box_images, project=True)
        box_text_feats = self.encode_text(box_tokens, project=True)
        all_image_feats = torch.cat(dist.gather_across_processes(image_feats), dim=0)
        all_text_feats = torch.cat(dist.gather_across_processes(text_feats), dim=0)

        with torch.autocast(self.device.type, dtype=torch.float32):
            image_scores, text_scores, image_vals, text_vals = self._main_scores(
                image_feats, text_feats, all_image_feats, all_text_feats, curv
            )
            box_image_scores, box_text_scores, _, _ = self._main_scores(
                box_image_feats, box_text_feats, all_image_feats, all_text_feats, curv
            )
            batch_size = image_feats.shape[0]
            targets = shifted_targets(batch_size, self._rank, image_scores.device)
            full_loss = image_text_loss(
                self.loss_type,
                image_scores,
                text_scores,
                targets,
                scale,
                bias=self.logit_bias if self.loss_type == "sigmoid" else None,
                sigmoid_normalization=self.sigmoid_normalization,
                sigmoid_negatives_per_sample=self.sigmoid_negatives_per_sample,
            )
            box_loss = image_text_loss(
                self.loss_type,
                box_image_scores,
                box_text_scores,
                targets,
                scale,
                bias=self.logit_bias if self.loss_type == "sigmoid" else None,
                sigmoid_normalization=self.sigmoid_normalization,
                sigmoid_negatives_per_sample=self.sigmoid_negatives_per_sample,
            )
            alignment_loss = 0.5 * (full_loss + box_loss)

            # Keep HyCoCLIP's explicit hierarchy as a compatibility baseline.
            text_image_ent = torch.clamp(
                compute_exterior_angle(text_feats, image_feats, curv, eps=self.lorentz_eps)
                - 0.7 * L.half_aperture(text_feats, curv, eps=self.lorentz_eps),
                min=0,
            ).mean()
            box_text_image_ent = torch.clamp(
                compute_exterior_angle(box_text_feats, box_image_feats, curv, eps=self.lorentz_eps)
                - 0.7 * L.half_aperture(box_text_feats, curv, eps=self.lorentz_eps),
                min=0,
            ).mean()
            cross_image_ent = torch.clamp(
                compute_exterior_angle(box_image_feats, image_feats, curv, eps=self.lorentz_eps)
                - 1.2 * L.half_aperture(box_image_feats, curv, eps=self.lorentz_eps),
                min=0,
            ).mean()
            cross_text_ent = torch.clamp(
                compute_exterior_angle(box_text_feats, text_feats, curv, eps=self.lorentz_eps)
                - 1.2 * L.half_aperture(box_text_feats, curv, eps=self.lorentz_eps),
                min=0,
            ).mean()
            entailment_loss = 0.5 * (text_image_ent + box_text_image_ent + cross_image_ent + cross_text_ent)
            total_loss = alignment_loss + (self.entail_weight * entailment_loss if self.entail_weight > 0 else 0)

        row_ids = torch.arange(batch_size, device=image_scores.device)
        logging = {
            "loss_main": alignment_loss.detach(),
            "alignment_loss": alignment_loss.detach(),
            f"{self.loss_type}_loss": alignment_loss.detach(),
            "image_text_alignment_loss": full_loss.detach(),
            "box_alignment_loss": box_loss.detach(),
            "entailment_loss": entailment_loss.detach(),
            "text_image_entailment_loss": text_image_ent.detach(),
            "box_text_image_entailment_loss": box_text_image_ent.detach(),
            "cross_image_entailment_loss": cross_image_ent.detach(),
            "cross_text_entailment_loss": cross_text_ent.detach(),
            "main_score_pos_mean": 0.5 * (image_scores[row_ids, targets].detach().mean() + text_scores[row_ids, targets].detach().mean()),
            "main_score_neg_mean": 0.5 * (image_scores.detach().mean() + text_scores.detach().mean()),
            "main_logit_scale": scale.detach(),
            "logit_scale": scale.detach(),
            "curv": curv.detach(),
        }
        if self.loss_type == "sigmoid":
            logging["main_logit_bias"] = self.logit_bias.detach()
            logging["logit_bias"] = self.logit_bias.detach()
        return {"loss": total_loss, "logging": logging}


class PHyCLIP(MERU):
    """Compatibility wrapper. Use the original PHyCLIP branch for exact PHyCLIP reproduction."""

    def __init__(
        self,
        visual: nn.Module,
        textual: TransformerTextEncoder,
        embed_dim: int,
        subspace_dim: int = 8,
        product_metric: str = "l1",
        **kwargs: object,
    ):
        self.subspace_dim = int(subspace_dim)
        self.product_metric = str(product_metric)
        super().__init__(visual=visual, textual=textual, embed_dim=embed_dim, **kwargs)
