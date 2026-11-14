import torch
from torch.nn import functional as F

from sigmeru import lorentz as L
from sigmeru.models import CLIPBaseline, HyCoCLIP, MERU, PHyCLIP


def process_text_features(
    model: PHyCLIP | HyCoCLIP | MERU | CLIPBaseline,
    text_features: torch.Tensor,
) -> torch.Tensor:
    """Convert already-encoded per-prompt text features into one class vector.

    This intentionally follows the original PHyCLIP/MERU zero-shot protocol:
    classification.py calls ``model.encode_text(tokens, project=False)`` for all
    prompt templates of one class, and this function performs prompt ensembling
    before the final geometry-specific projection/normalization.
    """
    if isinstance(model, PHyCLIP):
        # Map class features to each concept's hyperboloid after prompt averaging.
        class_feats = text_features.mean(dim=0).view(-1, model.num_subspaces, model.subspace_dim)
        class_feats = class_feats * torch.stack(
            [alpha.exp() for alpha in model.textual_concept_alphas]
        ).view(1, -1, 1)
        curvs = torch.stack([curv.exp() for curv in model.curvs])
        return L.exp_map0_batch(class_feats, curvs).view(model.num_subspaces, -1)

    if isinstance(model, (HyCoCLIP, MERU)):
        # Ensemble prompts in the tangent/pre-exp space, then project once to the
        # Lorentz manifold. This is the same classifier construction used by the
        # original PHyCLIP/MERU evaluation path.
        class_feats = text_features.mean(dim=0)
        class_feats = class_feats * model.textual_alpha.exp()
        return L.exp_map0(class_feats, model.curv.exp())

    # CLIP-style Euclidean prompt ensemble.
    class_feats = F.normalize(text_features, dim=-1).mean(dim=0)
    return F.normalize(class_feats, dim=-1)


def compute_similarity_scores(
    model: PHyCLIP | HyCoCLIP | MERU | CLIPBaseline,
    image_features: torch.Tensor,
    text_features: torch.Tensor,
) -> torch.Tensor:
    """Compute zero-shot scores.

    The classifier construction stays identical to the original PHyCLIP protocol.
    For newer SigMERU/MERU variants, the only intended change is here: models may
    expose ``compute_eval_similarity`` so ImageNet scoring follows their configured
    eval/main score type (e.g. root_angle or Lorentz distance). Legacy models that
    do not expose this method fall back to the original PHyCLIP/MERU scoring.
    """
    if hasattr(model, "compute_eval_similarity"):
        return model.compute_eval_similarity(image_features, text_features)  # type: ignore[attr-defined]

    if isinstance(model, PHyCLIP):
        curvs = torch.stack([curv.exp() for curv in model.curvs])
        image_features = image_features.view(-1, model.num_subspaces, model.subspace_dim)
        text_features = text_features.view(-1, model.num_subspaces, model.subspace_dim)
        return L.pairwise_inner_batch(image_features, text_features, curvs).sum(dim=0)

    if isinstance(model, (HyCoCLIP, MERU)):
        return L.pairwise_inner(image_features, text_features, model.curv.exp())

    return image_features @ text_features.transpose(0, 1)
