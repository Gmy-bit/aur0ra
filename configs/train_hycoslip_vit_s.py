"""AURORA ViT-S/16 training configuration.

The historical ``hycoslip`` model-factory name selects the MERU-family
implementation used by the AURORA training path in this code snapshot.
The AURORA recipe adds grounded box/phrase augmentation, root-angle alignment,
radial-shadow (RAOC/HEC) supervision, and ARGENT-space norm regularization.
"""

from .train_sigmeru_vit_l import dataset, model, optim, train  # noqa: F401

# ViT-S/16 visual backbone.
model.visual.arch = "vit_small_mocov3_patch16_224"

# AURORA training route. Do not replace ``hycoslip`` with ``hycoclip``:
# the latter selects the separate HyCoCLIP compatibility class.
dataset.mapper.use_boxes = True
dataset.drop_unused_box_fields = False
model.model_name = "hycoslip"
model.loss_type = "sigmoid"
model.use_boxes = True

# Main cross-modal alignment.
model.main_score_type = "root_angle"
model.eval_score_type = "main"
model.sigmoid_logit_bias_init = -6.5

# RAOC / HEC implementation branch.
model.entailment_loss_type = "radial_shadow"
model.entail_weight = 0.15
model.angular_temperature_init = 0.07
model.angular_logit_bias_init = -6.5
model.angular_reduction = "pair_mean"
model.angular_negative_mining = "all"
model.angular_topk = 0
model.radial_shadow_slope = 8.0
model.radial_shadow_radial_scale = 0.1
model.radial_shadow_radial_weight = 1.0

# ARGENT-space norm regularization.
model.norm_reg_type = "argent_space"
model.norm_weight = 0.05
