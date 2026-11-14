#!/usr/bin/env python
"""Visualize reference-set Lorentz geometry.

Top-of-file defaults are intentionally easy to edit for quick server runs.
HoroPCA and CO-SNE here are practical diagnostics:
  * HoroPCA-style: log/Poincare coordinates + PCA, not a full HoroPCA optimizer.
  * CO-SNE-style: t-SNE over Lorentz distance matrix when sklearn is available.
Angle/aperture/inside labels are always computed from the original Lorentz
metrics saved by extract_reference_embeddings.py, not from the 2D picture.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch

# Editable defaults.
REF_EMB_PATH = "./reference_outputs/reference_embeddings.pt"
OUT_DIR = "./reference_outputs/figures"
PLOT_RADIUS_DIST = True
PLOT_SPACE_NORM_DIST = True
PLOT_POINCARE_NORM_DIST = True
PLOT_HOROPCA = True
PLOT_COSNE = True
PLOT_PAIR_CONES = True
NUM_ANNOTATE = 10
PAIR_INDICES = [0, 1, 2, 3, 4]
SAVE_DPI = 300


def poincare_from_space(x: torch.Tensor, curv: float) -> torch.Tensor:
    c = torch.tensor(curv, dtype=x.dtype, device=x.device)
    time = torch.sqrt(torch.clamp(1.0 / c + torch.sum(x * x, dim=-1, keepdim=True), min=1e-30))
    radius = 1.0 / torch.sqrt(c)
    return x / (time + radius)


def pca2(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    x = x - x.mean(dim=0, keepdim=True)
    try:
        _, _, v = torch.pca_lowrank(x, q=2)
        return x @ v[:, :2]
    except Exception:
        _, _, vh = torch.linalg.svd(x, full_matrices=False)
        return x @ vh[:2].T


def save_hist(items: list[tuple[np.ndarray, str]], path: Path, title: str, xlabel: str, bins: int = 50):
    plt.figure(figsize=(7, 4.5))
    for arr, label in items:
        arr = np.asarray(arr)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        plt.hist(arr, bins=bins, alpha=0.45, density=True, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=SAVE_DPI)
    plt.close()


def plot_distributions(data: dict, out: Path):
    text_radius = data["text_radius"].numpy()
    image_radius = data["image_radius"].numpy()
    text_space = data["text_space_norm"].numpy()
    image_space = data["image_space_norm"].numpy()
    curv = float(data["curv"])
    text_p = poincare_from_space(data["text_lorentz"], curv)
    image_p = poincare_from_space(data["image_lorentz"], curv)
    text_p_norm = torch.norm(text_p, dim=-1).numpy()
    image_p_norm = torch.norm(image_p, dim=-1).numpy()

    if PLOT_RADIUS_DIST:
        save_hist([(text_radius, "text"), (image_radius, "image")], out / "radius_distribution.png", "Hyperbolic radius distribution", "radius")
    if PLOT_SPACE_NORM_DIST:
        save_hist([(text_space, "text"), (image_space, "image")], out / "space_norm_distribution.png", "Lorentz space norm distribution", "||space||")
    if PLOT_POINCARE_NORM_DIST:
        save_hist([(text_p_norm, "text"), (image_p_norm, "image")], out / "poincare_norm_distribution.png", "Poincare norm distribution", "||poincare||")

    pos_dist = data["pairwise_lorentz_distance"].diag().numpy()
    pos_rapid = data["pairwise_rapidity"].diag().numpy()
    pos_alpha = data["positive_alpha"].numpy()
    pos_cos = data["positive_cos_alpha"].numpy()
    save_hist([(pos_dist, "positive")], out / "positive_distance_distribution.png", "Positive Lorentz distance", "d_L")
    save_hist([(pos_rapid, "positive")], out / "positive_rapidity_distribution.png", "Positive rapidity", "rho")
    if "positive_root_angle" in data:
        save_hist([(data["positive_root_angle"].numpy(), "positive")], out / "positive_root_angle_distribution.png", "Positive root angle", "root angle")
    if "positive_root_angle_cos" in data:
        save_hist([(data["positive_root_angle_cos"].numpy(), "positive")], out / "positive_root_angle_cos_distribution.png", "Positive cos(root angle)", "cos(root angle)")
    save_hist([(pos_alpha, "positive")], out / "positive_alpha_distribution.png", "Positive exterior angle", "alpha")
    save_hist([(pos_cos, "positive")], out / "positive_cos_alpha_distribution.png", "Positive cos alpha", "cos(alpha)")


def plot_horopca(data: dict, out: Path, num_annotate: int):
    curv = float(data["curv"])
    text_p = poincare_from_space(data["text_lorentz"], curv)
    image_p = poincare_from_space(data["image_lorentz"], curv)
    combined = torch.cat([text_p, image_p], dim=0)
    xy = pca2(combined).numpy()
    n = text_p.shape[0]
    text_xy, image_xy = xy[:n], xy[n:]
    radii = torch.cat([data["text_radius"], data["image_radius"]]).numpy()

    plt.figure(figsize=(8, 8))
    plt.scatter(text_xy[:, 0], text_xy[:, 1], s=18 + 10 * data["text_radius"].numpy(), alpha=0.65, label="text", marker="^")
    plt.scatter(image_xy[:, 0], image_xy[:, 1], s=18 + 10 * data["image_radius"].numpy(), alpha=0.55, label="image", marker="o")
    inside = data["positive_inside_cone"].numpy().astype(bool)
    for i in range(min(n, 40)):
        color = "tab:green" if inside[i] else "tab:red"
        plt.plot([text_xy[i, 0], image_xy[i, 0]], [text_xy[i, 1], image_xy[i, 1]], color=color, alpha=0.35, linewidth=0.8)
    for i in range(min(num_annotate, n)):
        plt.text(text_xy[i, 0], text_xy[i, 1], f"t{i}", fontsize=8)
        plt.text(image_xy[i, 0], image_xy[i, 1], f"v{i}", fontsize=8)
    plt.title("Practical HoroPCA-style visualization (Poincare/log coordinates + PCA)")
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(out / "horopca_reference.png", dpi=SAVE_DPI)
    plt.close()


def plot_cosne(data: dict, out: Path, num_annotate: int):
    text = data["text_lorentz"]
    image = data["image_lorentz"]
    n = text.shape[0]
    # Block distance matrix: [text,image] x [text,image].
    tt = torch.cdist(text.float(), text.float()).numpy()
    ii = torch.cdist(image.float(), image.float()).numpy()
    ti = data["pairwise_lorentz_distance"].numpy()
    dist = np.block([[tt, ti], [ti.T, ii]])
    try:
        from sklearn.manifold import TSNE
        perplexity = max(5, min(30, (2 * n - 1) // 3))
        xy = TSNE(n_components=2, metric="precomputed", init="random", perplexity=perplexity, learning_rate="auto", random_state=0).fit_transform(dist)
        title = "CO-SNE-style t-SNE from Lorentz distance matrix"
    except Exception:
        xy = pca2(torch.cat([text, image], dim=0)).numpy()
        title = "CO-SNE fallback: PCA because sklearn TSNE is unavailable"
    text_xy, image_xy = xy[:n], xy[n:]
    inside = data["positive_inside_cone"].numpy().astype(bool)
    plt.figure(figsize=(8, 8))
    plt.scatter(text_xy[:, 0], text_xy[:, 1], s=24, alpha=0.65, label="text", marker="^")
    plt.scatter(image_xy[:, 0], image_xy[:, 1], s=24, alpha=0.55, label="image", marker="o")
    for i in range(min(n, 40)):
        color = "tab:green" if inside[i] else "tab:red"
        plt.plot([text_xy[i, 0], image_xy[i, 0]], [text_xy[i, 1], image_xy[i, 1]], color=color, alpha=0.35, linewidth=0.8)
    for i in range(min(num_annotate, n)):
        plt.text(text_xy[i, 0], text_xy[i, 1], f"t{i}", fontsize=8)
        plt.text(image_xy[i, 0], image_xy[i, 1], f"v{i}", fontsize=8)
    plt.title(title)
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(out / "cosne_reference.png", dpi=SAVE_DPI)
    plt.close()


def plot_pair_cones(data: dict, out: Path, pair_indices: Iterable[int]):
    curv = float(data["curv"])
    text_p = poincare_from_space(data["text_lorentz"], curv).numpy()
    image_p = poincare_from_space(data["image_lorentz"], curv).numpy()
    alpha = data["positive_alpha"].numpy()
    cos_alpha = data["positive_cos_alpha"].numpy()
    aperture = data["positive_aperture"].numpy()
    ext_minus = data["positive_ext_minus_aper"].numpy()
    inside = data["positive_inside_cone"].numpy().astype(bool)
    pos_dist = data["pairwise_lorentz_distance"].diag().numpy()
    pos_rapid = data["pairwise_rapidity"].diag().numpy()
    pos_root_angle = data.get("positive_root_angle", torch.full_like(data["positive_alpha"], float("nan"))).numpy()
    pos_root_cos = data.get("positive_root_angle_cos", torch.full_like(data["positive_cos_alpha"], float("nan"))).numpy()
    tr = data["text_radius"].numpy()
    ir = data["image_radius"].numpy()

    # We plot the 2D plane spanned by text and image spatial components via PCA.
    # The cone wedge is a Poincare-disk visual approximation; numeric alpha and
    # inside decisions come from Lorentz formulas above.
    for i in pair_indices:
        if i < 0 or i >= len(alpha):
            continue
        pts = torch.tensor(np.stack([np.zeros_like(text_p[i]), text_p[i], image_p[i]], axis=0), dtype=torch.float32)
        xy = pca2(pts).numpy()
        origin, tx, im = xy[0], xy[1], xy[2]
        color = "tab:green" if inside[i] else "tab:red"
        plt.figure(figsize=(6, 6))
        lim = max(1e-3, np.abs(xy).max() * 1.4)
        circle = plt.Circle((0, 0), lim, color="lightgray", alpha=0.12)
        ax = plt.gca(); ax.add_patch(circle)
        plt.scatter([origin[0]], [origin[1]], c="black", s=40, label="origin")
        plt.scatter([tx[0]], [tx[1]], c="tab:green", marker="^", s=80, label="text parent")
        plt.scatter([im[0]], [im[1]], c="tab:blue", marker="o", s=80, label="image child")
        plt.plot([origin[0], tx[0]], [origin[1], tx[1]], color="gray", linestyle="--", linewidth=1)
        plt.arrow(tx[0], tx[1], im[0]-tx[0], im[1]-tx[1], length_includes_head=True, head_width=0.02*lim, color=color, alpha=0.75)
        plt.plot([tx[0], im[0]], [tx[1], im[1]], color=color, linewidth=1.4)
        txt = (
            f"idx={i}\ninside={inside[i]}\n"
            f"root={pos_root_angle[i]:.3f}, root_cos={pos_root_cos[i]:.3f}\n"
            f"alpha={alpha[i]:.3f}, cos={cos_alpha[i]:.3f}\n"
            f"aperture={aperture[i]:.3f}, alpha-aper={ext_minus[i]:.3f}\n"
            f"d={pos_dist[i]:.3f}, rho={pos_rapid[i]:.3f}\n"
            f"r_text={tr[i]:.3f}, r_img={ir[i]:.3f}"
        )
        plt.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left", fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.75))
        plt.title("Pair cone diagnostic (Poincare/PCA visual approximation)")
        plt.legend(loc="lower right")
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig(out / f"pair_cone_{i:06d}.png", dpi=SAVE_DPI)
        plt.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--emb_path", default=REF_EMB_PATH)
    p.add_argument("--out_dir", default=OUT_DIR)
    p.add_argument("--num_annotate", type=int, default=NUM_ANNOTATE)
    p.add_argument("--pair_indices", default=",".join(map(str, PAIR_INDICES)))
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = torch.load(args.emb_path, map_location="cpu")

    plot_distributions(data, out)
    if PLOT_HOROPCA:
        plot_horopca(data, out, args.num_annotate)
    if PLOT_COSNE:
        plot_cosne(data, out, args.num_annotate)
    if PLOT_PAIR_CONES:
        pair_indices = [int(x) for x in args.pair_indices.split(",") if x.strip()]
        plot_pair_cones(data, out, pair_indices)
    print(f"[DONE] figures saved to {out}")


if __name__ == "__main__":
    main()
