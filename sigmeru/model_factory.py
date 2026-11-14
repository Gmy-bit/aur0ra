from __future__ import annotations

from torch import nn

from sigmeru.encoders.text_encoders import TransformerTextEncoder
from sigmeru.models import CLIPBaseline, HyCoCLIP, MERU, PHyCLIP


def build_image_text_model(
    visual: nn.Module,
    textual: TransformerTextEncoder,
    embed_dim: int,
    model_name: str = "meru",
    loss_type: str = "sigmoid",
    use_boxes: bool = False,
    curv_init: float = 1.0,
    learn_curv: bool = True,
    curv_min: float | None = 0.1,
    curv_max: float | None = 10.0,
    entail_weight: float = 0.0,
    lorentz_eps: float = 1e-8,
    subspace_dim: int = 8,
    product_metric: str = "l1",
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
    pixel_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    pixel_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    **deprecated_kwargs: object,
) -> nn.Module:
    """Unified factory for CLIP / MERU / SigMERU / HyCoCLIP / PHyCLIP.

    ``deprecated_kwargs`` intentionally absorbs removed v16-v29 experiment knobs
    so old YAMLs fail less often during evaluation, but those knobs are ignored.
    """
    model_name = str(model_name).lower().strip()

    if model_name == "clip":
        return CLIPBaseline(
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

    common_hyper_kwargs = dict(
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
        eval_score_type=eval_score_type,
        entailment_loss_type=entailment_loss_type,
        angular_temperature_init=angular_temperature_init,
        angular_logit_scale_init=angular_logit_scale_init,
        angular_logit_bias_init=angular_logit_bias_init,
        angular_reduction=angular_reduction,
        angular_negative_mining=angular_negative_mining,
        angular_topk=angular_topk,
        norm_reg_type=norm_reg_type,
        norm_weight=norm_weight,
        norm_reg_eps=norm_reg_eps,
        norm_reg_alpha=norm_reg_alpha,
        text_feature_dropout=text_feature_dropout,
        disp_weight=disp_weight,
        disp_scale=disp_scale,
        disp_bias=disp_bias,
        radial_shadow_slope=radial_shadow_slope,
        radial_shadow_radial_scale=radial_shadow_radial_scale,
        radial_shadow_radial_weight=radial_shadow_radial_weight,
        shadow_overlap_weight=shadow_overlap_weight,
        shadow_overlap_scale=shadow_overlap_scale,
        shadow_overlap_bias=shadow_overlap_bias,
    )

    if model_name in {"meru", "hycoslip", "sigmeru"}:
        return MERU(**common_hyper_kwargs)
    if model_name == "hycoclip":
        return HyCoCLIP(**common_hyper_kwargs)
    if model_name == "phyclip":
        return PHyCLIP(
            **common_hyper_kwargs,
            subspace_dim=subspace_dim,
            product_metric=product_metric,
        )

    raise ValueError(
        "model_name must be one of ['clip', 'meru', 'hycoslip', 'sigmeru', 'hycoclip', 'phyclip'], "
        f"got {model_name!r}."
    )
