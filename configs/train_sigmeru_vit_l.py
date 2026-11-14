# ---------------------------------------
# Unified training config for CLIP / MERU / SigMERU / HyCoCLIP / PHyCLIP.
# v31 default: root-angle SigMERU route.
# ---------------------------------------

from torch.optim.adamw import AdamW
from torchvision import transforms as T

from sigmeru.config import LazyCall as L
from sigmeru.data.webdataset_mapper import GroundedDatasetTarMapper, ImageTextWebDataset
from sigmeru.encoders.image_encoders import build_timm_vit
from sigmeru.encoders.text_encoders import TransformerTextEncoder
from sigmeru.model_factory import build_image_text_model
from sigmeru.optim import LinearWarmupCosineDecayLR, set_weight_decay_per_param


dataset = L(ImageTextWebDataset)(
    tarfiles=["datasets/train/GRIT/processed/*.tar"],
    mapper=L(GroundedDatasetTarMapper)(
        image_transform=[
            L(T.RandomResizedCrop)(
                size=224, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC
            ),
            L(T.ToTensor)(),
        ],
        use_boxes=False,
    ),
    buffer_size=4000,
    seed="${..train.seed}",
    drop_unused_box_fields=True,
)


model = L(build_image_text_model)(
    model_name="meru",          # clip | meru | hycoclip | phyclip
    loss_type="sigmoid",       # contrastive | sigmoid
    use_boxes=False,

    visual=L(build_timm_vit)(
        arch="vit_large_patch16_224",
        global_pool="token",
        use_sincos2d_pos=True,
    ),
    textual=L(TransformerTextEncoder)(
        arch="L12_W512",
        vocab_size=49408,
        context_length=77,
    ),
    embed_dim=512,

    # Clean hyperbolic geometry defaults.
    curv_init=1.0,
    learn_curv=True,
    curv_min=0.1,
    curv_max=10.0,
    lorentz_eps=1e-8,

    # Main alignment score for MERU/SigMERU.
    #   root_angle      : score = cos ∠xOy at the Lorentz origin O
    #   lorentz_distance: score = -d_L(x,y)
    #   rapidity        : score = -rho(x,y), rho = acosh(-c<x,y>_L)
    main_score_type="root_angle",
    eval_score_type="main",    # main | root_angle | lorentz_distance | rapidity | angular_cos | typed_shadow | radial_shadow

    # SigLIP-style pairwise sigmoid controls.
    # B=768 bounded-cosine prior: -log(B-1)≈-6.64, so -6.5 is a reasonable manual default.
    # Distance/rapidity ablations are unbounded and usually need a manually-set positive bias, e.g. 10.0.
    sigmoid_logit_bias_init=-6.5,
    sigmoid_normalization="batch",
    sigmoid_negatives_per_sample=0,

    # Angular entailment.  Always asymmetric text(parent)->image(child).
    #   none         : no angular entailment
    #   positive_cos : mean_i [1 - cos alpha(t_i, v_i)]
    #   contrastive_cos : CLIP-like T2I CE loss on cos alpha(t_i, v_j)
    #   sigmoid_cos     : SigLIP-like pairwise loss on cos alpha(t_i, v_j)
    #   typed_shadow    : legacy psi(text)-root_angle boundary shadow depth
    #   radial_shadow   : legacy joint-min surrogate for the typed radial+shadow order
    #   radial_shadow_decomposed: pairwise shadow membership + positive radial type order
    entailment_loss_type="none",
    entail_weight=0.0,
    # T2I exterior-angle contrastive/sigmoid loss temperature. 0.07 => scale=1/0.07.
    angular_temperature_init=0.07,
    angular_logit_scale_init=10.0,  # fallback only when angular_temperature_init=None
    # B=768 sigmoid_cos prior: -log(B-1)≈-6.64. For a stricter geometric threshold
    # near cos(alpha)=0.7 with tau=0.07, manually try -10.0.
    angular_logit_bias_init=-6.5,
    angular_reduction="pair_mean",       # pair_mean | balanced
    angular_negative_mining="all",       # all | topk
    angular_topk=0,                      # active only when angular_negative_mining="topk"

    # Minimal text augmentation in Euclidean/tangent feature space:
    # Text Encoder -> textual_proj -> dropout -> exp_map0. Disabled by default.
    text_feature_dropout=0.0,

    # Intra-modal directional dispersive regularizer. Disabled by default.
    disp_weight=0.0,
    disp_scale=10.0,
    disp_bias=-1.0,

    # Modality-Typed Radial Shadow.
    # Allowed angular shadow: min(pi, slope * relu(rho_image-rho_text)).
    # radial_shadow keeps the R1 joint-min baseline; radial_shadow_decomposed
    # trains angular membership and positive-only radial type order separately.
    radial_shadow_slope=8.0,
    radial_shadow_radial_scale=0.1,
    radial_shadow_radial_weight=1.0,

    # Directed anti-containment regularization between local text shadows.
    # Ordinary cap overlap is allowed; only erroneous B_j subset B_i is penalized.
    # Apertures are detached and weight=0 fully skips the local text-text matrix.
    shadow_overlap_weight=0.0,
    shadow_overlap_scale=10.0,
    shadow_overlap_bias=-1.0,

    # Norm regularization.
    #   none
    #   argent_space / argent_radius: old ARGENT-style t^2-log(t)
    #   inverse / inverse_space / inverse_radius: 1/(norm+eps)
    #   inverse_square / inverse_square_space / inverse_square_radius: 1/(norm+eps)^2
    #   softplus_neg / softplus_neg_space / softplus_neg_radius: softplus(-alpha*norm)
    norm_reg_type="none",
    norm_weight=0.0,
    norm_reg_eps=1e-6,
    norm_reg_alpha=5.0,

    # PHyCLIP-only knobs. Ignored by clip/meru/hycoclip branches.
    subspace_dim=8,
    product_metric="l1",
)


optim = dict(
    optimizer=L(AdamW)(
        params=L(set_weight_decay_per_param)(
            weight_decay="${..weight_decay}",
            gain_bias_decay=0.0,
            exclude_params=[
                "logit_scale",
                "logit_bias",
                "angular_logit_scale",
                "angular_logit_bias",
                "visual_alpha",
                "textual_alpha",
                "curv",
                "curvs",
                "visual_concept_alphas",
                "textual_concept_alphas",
            ],
        ),
        lr=5e-4,
        betas=(0.9, 0.98),
        weight_decay=0.2,
        fused=False,
    ),
    lr_scheduler=L(LinearWarmupCosineDecayLR)(
        total_steps="${...train.num_iterations}", warmup_steps=4000
    ),
)


train = dict(
    seed=0,
    amp=True,
    total_batch_size=768,
    num_iterations=500000,
    cudnn_benchmark=True,
    cudnn_deterministic=False,
    num_workers=4,
    non_blocking_to_device=True,
    zero_grad_set_to_none=True,
    sync_cuda_timing=False,
    log_all_ranks_to_stderr=False,
    prefetch_factor=2,
    persistent_workers=False,
    experiment_tag="",

    compile_model=False,
    compile_backend="inductor",
    compile_mode="default",
    compile_fullgraph=False,
    compile_dynamic=None,
    compile_ddp_static_graph=False,
    compile_find_unused_parameters=True,

    use_tf32=True,
    float32_matmul_precision="high",
    cuda_matmul_tf32=True,
    cudnn_tf32=True,

    ddp=dict(
        broadcast_buffers=False,
        static_graph=False,
        find_unused_parameters=False,
        gradient_as_bucket_view=False,
    ),
    ddp_fp16_compression=True,
    run_eval=False,
)
