# ---------------------------------------
# HyCoCLIP training config through the unified model factory.
# This keeps scripts/run_train_sigmeru.sh overrides compatible with HyCoCLIP.
# ---------------------------------------

from .train_sigmeru_vit_l import dataset, model, optim, train  # noqa: F401

# HyCoCLIP uses localized image/text boxes.
dataset.mapper.use_boxes = True
dataset.drop_unused_box_fields = False

model.model_name = "hycoclip"
model.loss_type = "contrastive"
model.use_boxes = True
model.entail_weight = 0.2
model.sigmoid_logit_mode = "distance_margin"
model.sigmoid_logit_bias_init = 1.0

# Original HyCoCLIP/MERU style settings.
model.learn_curv = True
model.curv_init = 1.0
model.curv_min = 0.1
model.curv_max = 10.0

# AdamW scalar exclusions stay inherited from train_sigmeru_vit_l.
