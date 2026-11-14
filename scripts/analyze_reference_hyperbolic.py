#!/usr/bin/env python
"""Analyze a small SigMERU/MERU reference set in hyperbolic space.

This script is designed for diagnostic visualization rather than exact paper-
faithful HoroPCA/CO-SNE reproduction.  It provides:
  * reference embedding extraction
  * radius / Poincare norm / Lorentz distance distributions
  * HoroPCA-like PCA in tangent/Poincare coordinates
  * CO-SNE-like t-SNE using pairwise hyperbolic distances when sklearn exists
  * local 2D cone plots for selected text-image pairs
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sigmeru import lorentz as L
from sigmeru.config import LazyConfig, LazyFactory
from sigmeru.tokenizer import Tokenizer
from sigmeru.utils.checkpointing import CheckpointManager

# -----------------------------------------------------------------------------
# Editable plotting switches.
# -----------------------------------------------------------------------------
PLOT_RADIUS_DISTRIBUTION = True
PLOT_POINCARE_NORM_DISTRIBUTION = True
PLOT_DISTANCE_DISTRIBUTION = True
PLOT_HOROPCA = True
PLOT_COSNE = True
PLOT_LOCAL_CONES = True
DEFAULT_NUM_CONE_PAIRS = 16


class ReferenceDataset(Dataset):
    def __init__(self, root: str | Path, image_size: int = 224, include_boxes: bool = True, max_samples: int = 0):
        self.root = Path(root)
        meta_path = self.root / "metadata.jsonl"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata.jsonl not found in {self.root}")
        self.records = [json.loads(line) for line in meta_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if max_samples and max_samples > 0:
            self.records = self.records[:max_samples]
        self.include_boxes = bool(include_boxes)
        self.transform = T.Compose([
            T.Resize(image_size, T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, rel: str) -> torch.Tensor:
        img = Image.open(self.root / rel).convert("RGB")
        return self.transform(img)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        out = {
            "id": int(r.get("id", idx)),
            "image": self._load_image(r["image"]),
            "caption": r.get("caption", ""),
            "has_box": bool(r.get("has_box", False) and self.include_boxes),
            "box_caption": r.get("box_caption", ""),
        }
        if out["has_box"] and "box_image" in r:
            out["box_image"] = self._load_image(r["box_image"])
        else:
            out["box_image"] = torch.zeros_like(out["image"])
        return out


def collate_fn(batch):
    return {
        "ids": torch.tensor([b["id"] for b in batch], dtype=torch.long),
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "caption": [b["caption"] for b in batch],
        "has_box": torch.tensor([b["has_box"] for b in batch], dtype=torch.bool),
        "box_image": torch.stack([b["box_image"] for b in batch], dim=0),
        "box_caption": [b["box_caption"] for b in batch],
    }


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def load_model(config_path: str, checkpoint_path: str, device: torch.device):
    cfg = LazyConfig.load(config_path)
    model = LazyFactory.build_model(cfg, device).eval()
    manager = CheckpointManager(Path(checkpoint_path).parent, model=model)
    manager.load(checkpoint_path)
    return model


@torch.inference_mode()
def encode_reference(model, dataset: ReferenceDataset, batch_size: int, num_workers: int, device: torch.device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    tokenizer = Tokenizer()
    all_ids, all_img, all_txt, all_box_img, all_box_txt, all_has_box = [], [], [], [], [], []

    for batch in tqdm(loader, desc="Encoding reference set"):
        images = batch["image"].to(device, non_blocking=True)
        tokens = tokenizer(batch["caption"])
        model_u = unwrap(model)

        # Hyperbolic embeddings are always needed for entailment/radius analysis.
        image_h = model_u.encode_image(images, project=True)
        text_h = model_u.encode_text(tokens, project=True)
        all_ids.append(batch["ids"])
        all_img.append(image_h.detach().cpu())
        all_txt.append(text_h.detach().cpu())
        all_has_box.append(batch["has_box"])

        if batch["has_box"].any():
            box_images = batch["box_image"].to(device, non_blocking=True)
            # Empty box captions fall back to the full caption to keep tensor shapes simple.
            box_caps = [bc if bc else c for bc, c in zip(batch["box_caption"], batch["caption"])]
            box_tokens = tokenizer(box_caps)
            box_image_h = model_u.encode_image(box_images, project=True)
            box_text_h = model_u.encode_text(box_tokens, project=True)
            all_box_img.append(box_image_h.detach().cpu())
            all_box_txt.append(box_text_h.detach().cpu())
        else:
            all_box_img.append(torch.empty(0))
            all_box_txt.append(torch.empty(0))

    out = {
        "ids": torch.cat(all_ids, dim=0),
        "image": torch.cat(all_img, dim=0),
        "text": torch.cat(all_txt, dim=0),
        "has_box": torch.cat(all_has_box, dim=0),
    }
    if any(x.numel() > 0 for x in all_box_img):
        # Keep only entries that truly have boxes.
        box_images, box_texts = [], []
        for bi, bt in zip(all_box_img, all_box_txt):
            if bi.numel() > 0:
                box_images.append(bi.cpu())
                box_texts.append(bt.cpu())
        if box_images:
            out["box_image"] = torch.cat(box_images, dim=0)
            out["box_text"] = torch.cat(box_texts, dim=0)
    return out


def poincare_from_lorentz(x: torch.Tensor, curv: torch.Tensor) -> torch.Tensor:
    # x = [space, time].  Lorentz radius is 1/sqrt(c).  Project to Poincare ball.
    space = x[..., :-1]
    time = x[..., -1:].clamp_min(1e-6)
    radius = 1.0 / torch.sqrt(curv).clamp_min(1e-6)
    return space / (time + radius)


def pca2(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    x = x - x.mean(dim=0, keepdim=True)
    try:
        u, s, v = torch.pca_lowrank(x, q=2)
        return x @ v[:, :2]
    except Exception:
        _, _, vh = torch.linalg.svd(x, full_matrices=False)
        return x @ vh[:2].T


def save_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def compute_pair_metrics(image: torch.Tensor, text: torch.Tensor, curv: torch.Tensor):
    device = image.device
    dist_mat = L.pairwise_dist(image, text, curv)
    pos_dist = dist_mat.diag()
    image_radius = L.get_distance_from_origin(image, curv)
    text_radius = L.get_distance_from_origin(text, curv)
    angle = L.oxy_angle(text, image, curv)
    aperture = L.half_aperture(text, curv)
    score = aperture - angle
    return {
        "pos_dist": pos_dist,
        "image_radius": image_radius,
        "text_radius": text_radius,
        "exterior_angle": angle,
        "aperture": aperture,
        "entail_score": score,
        "inside": score > 0,
    }


def plot_hist(values, labels, path: Path, title: str, xlabel: str):
    plt.figure(figsize=(7, 4.5))
    for v, lab in zip(values, labels):
        if len(v) == 0:
            continue
        plt.hist(v, bins=50, alpha=0.45, density=True, label=lab)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_horopca(image: torch.Tensor, text: torch.Tensor, box_image: torch.Tensor | None, box_text: torch.Tensor | None, curv: torch.Tensor, metrics, out: Path):
    pcs = []
    labels = []
    colors = []
    p_img = poincare_from_lorentz(image, curv)
    p_txt = poincare_from_lorentz(text, curv)
    pcs.extend([p_img, p_txt])
    labels.extend(["Images", "Texts"])
    colors.extend(["tab:blue", "tab:green"])
    if box_image is not None and box_image.numel() > 0:
        pcs.append(poincare_from_lorentz(box_image, curv)); labels.append("Image boxes"); colors.append("tab:orange")
    if box_text is not None and box_text.numel() > 0:
        pcs.append(poincare_from_lorentz(box_text, curv)); labels.append("Text boxes"); colors.append("tab:red")
    all_p = torch.cat(pcs, dim=0)
    xy = pca2(all_p)
    plt.figure(figsize=(7, 7))
    start = 0
    for arr, lab, col in zip(pcs, labels, colors):
        n = arr.shape[0]
        plt.scatter(xy[start:start+n, 0], xy[start:start+n, 1], s=9, alpha=0.55, label=lab, c=col)
        start += n
    # connect first few image-text positives, green if inside, red if outside
    n_pairs = min(32, image.shape[0])
    img_xy = xy[: image.shape[0]]
    txt_xy = xy[image.shape[0]: image.shape[0] + text.shape[0]]
    inside = metrics["inside"].detach().cpu()
    for i in range(n_pairs):
        col = "tab:green" if bool(inside[i]) else "tab:red"
        plt.plot([txt_xy[i,0], img_xy[i,0]], [txt_xy[i,1], img_xy[i,1]], color=col, alpha=0.35, linewidth=0.8)
    plt.title("HoroPCA-like projection (Poincare coordinates + PCA)")
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(out / "horopca_pca2.png", dpi=200)
    plt.close()


def plot_cosne(image: torch.Tensor, text: torch.Tensor, curv: torch.Tensor, out: Path):
    combined = torch.cat([image, text], dim=0)
    try:
        from sklearn.manifold import TSNE
        dist = L.pairwise_dist(combined, combined, curv).detach().cpu().float().numpy()
        perplexity = min(30, max(5, (combined.shape[0] - 1) // 5))
        xy = TSNE(n_components=2, metric="precomputed", perplexity=perplexity, init="random", learning_rate="auto").fit_transform(dist)
        xy = torch.tensor(xy)
        title = "CO-SNE-like t-SNE on Lorentz distance"
    except Exception as e:
        print(f"[WARN] sklearn TSNE unavailable or failed ({e}); falling back to PCA.")
        xy = pca2(poincare_from_lorentz(combined, curv))
        title = "CO-SNE fallback: PCA on Poincare coordinates"
    n = image.shape[0]
    plt.figure(figsize=(7, 7))
    plt.scatter(xy[:n, 0], xy[:n, 1], s=9, alpha=0.55, label="Images")
    plt.scatter(xy[n:, 0], xy[n:, 1], s=9, alpha=0.55, label="Texts")
    plt.title(title)
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(out / "cosne_tsne2.png", dpi=200)
    plt.close()


def plot_local_cone_pair(i: int, image: torch.Tensor, text: torch.Tensor, curv: torch.Tensor, metrics, out: Path):
    # Use log_0 coordinates and the 2D subspace spanned by the text and image rays.
    t_vec = L.log_map0(text[i:i+1], curv).squeeze(0).detach().cpu().float()
    v_vec = L.log_map0(image[i:i+1], curv).squeeze(0).detach().cpu().float()
    e1 = t_vec / t_vec.norm().clamp_min(1e-8)
    residual = v_vec - (v_vec @ e1) * e1
    if residual.norm() < 1e-6:
        tmp = torch.zeros_like(e1); tmp[0] = 1.0
        residual = tmp - (tmp @ e1) * e1
    e2 = residual / residual.norm().clamp_min(1e-8)
    t_xy = torch.tensor([t_vec @ e1, t_vec @ e2])
    v_xy = torch.tensor([v_vec @ e1, v_vec @ e2])
    aper = float(metrics["aperture"][i])
    ext = float(metrics["exterior_angle"][i])
    score = float(metrics["entail_score"][i])
    inside = score >= 0

    plt.figure(figsize=(6, 6))
    plt.scatter([0], [0], c="black", s=40, marker="o", label="origin")
    plt.scatter([t_xy[0]], [t_xy[1]], c="tab:green", s=55, marker="^", label="text")
    plt.scatter([v_xy[0]], [v_xy[1]], c="tab:blue", s=55, marker="^", label="image")
    plt.plot([0, t_xy[0]], [0, t_xy[1]], c="gray", linestyle="--", linewidth=1)
    plt.plot([t_xy[0], v_xy[0]], [t_xy[1], v_xy[1]], c="tab:green" if inside else "tab:red", linewidth=1.2)

    # Cone wedge in the local tangent approximation: apex at text, axis away from origin.
    axis = t_xy / t_xy.norm().clamp_min(1e-8)
    angles = torch.linspace(-aper, aper, 80)
    length = max(float(v_xy.norm()), float(t_xy.norm())) * 0.55
    pts = []
    for a in angles:
        ca, sa = math.cos(float(a)), math.sin(float(a))
        rot = torch.tensor([axis[0] * ca - axis[1] * sa, axis[0] * sa + axis[1] * ca])
        pts.append(t_xy + length * rot)
    pts = torch.stack(pts)
    plt.plot(pts[:, 0], pts[:, 1], c="orange", linewidth=1.3, label="approx. cone")
    plt.fill_between(pts[:, 0], pts[:, 1], t_xy[1], color="orange", alpha=0.06)
    plt.title(f"pair {i}: {'inside' if inside else 'outside'} | ext={ext:.3f}, aper={aper:.3f}, ext-aper={ext-aper:.3f}")
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(out / f"cone_pair_{i:04d}.png", dpi=200)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference-dir", required=True)
    p.add_argument("--checkpoint-path", required=True)
    p.add_argument("--train-config", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-samples", type=int, default=500)
    p.add_argument("--include-boxes", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--num-cone-pairs", type=int, default=DEFAULT_NUM_CONE_PAIRS)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dataset = ReferenceDataset(args.reference_dir, include_boxes=args.include_boxes, max_samples=args.max_samples)
    model = load_model(args.train_config, args.checkpoint_path, device).to(device).eval()
    model_u = unwrap(model)
    curv = model_u.curv.exp().detach().cpu() if hasattr(model_u, "curv") else torch.tensor(1.0)

    emb = encode_reference(model, dataset, args.batch_size, args.num_workers, device)
    torch.save({**emb, "curv": curv}, out / "reference_embeddings.pt")

    image = emb["image"].to(device)
    text = emb["text"].to(device)
    curv_d = curv.to(device)
    metrics = {k: v.detach().cpu() for k, v in compute_pair_metrics(image, text, curv_d).items()}

    rows = []
    for i, rid in enumerate(emb["ids"].tolist()):
        rows.append({
            "id": int(rid),
            "image_radius": float(metrics["image_radius"][i]),
            "text_radius": float(metrics["text_radius"][i]),
            "lorentz_distance": float(metrics["pos_dist"][i]),
            "exterior_angle": float(metrics["exterior_angle"][i]),
            "aperture": float(metrics["aperture"][i]),
            "entail_score": float(metrics["entail_score"][i]),
            "inside_cone": bool(metrics["inside"][i]),
        })
    save_csv(out / "reference_metrics.csv", rows)
    inside_ratio = float(metrics["inside"].float().mean())
    print(f"[INFO] inside_cone_ratio={inside_ratio:.4f}, curv={float(curv):.4f}, n={len(rows)}")

    box_image = emb.get("box_image")
    box_text = emb.get("box_text")
    if box_image is not None: box_image = box_image.to(device)
    if box_text is not None: box_text = box_text.to(device)

    if PLOT_RADIUS_DISTRIBUTION:
        vals = [metrics["image_radius"].numpy(), metrics["text_radius"].numpy()]
        labs = ["Images", "Texts"]
        if box_image is not None and box_image.numel() > 0:
            vals.append(L.get_distance_from_origin(box_image, curv_d).detach().cpu().numpy()); labs.append("Image boxes")
        if box_text is not None and box_text.numel() > 0:
            vals.append(L.get_distance_from_origin(box_text, curv_d).detach().cpu().numpy()); labs.append("Text boxes")
        plot_hist(vals, labs, out / "radius_distribution.png", "Hyperbolic radius distribution", "d(origin, point)")

    if PLOT_POINCARE_NORM_DISTRIBUTION:
        vals = [poincare_from_lorentz(image, curv_d).norm(dim=-1).detach().cpu().numpy(), poincare_from_lorentz(text, curv_d).norm(dim=-1).detach().cpu().numpy()]
        labs = ["Images", "Texts"]
        plot_hist(vals, labs, out / "poincare_norm_distribution.png", "Poincare norm distribution", "||p||")

    if PLOT_DISTANCE_DISTRIBUTION:
        plot_hist([metrics["pos_dist"].numpy()], ["positive image-text pairs"], out / "lorentz_distance_distribution.png", "Positive Lorentz distance distribution", "d(image, text)")

    if PLOT_HOROPCA:
        plot_horopca(image, text, box_image, box_text, curv_d, metrics, out)

    if PLOT_COSNE:
        plot_cosne(image, text, curv_d, out)

    if PLOT_LOCAL_CONES:
        n = min(args.num_cone_pairs, image.shape[0])
        for i in range(n):
            plot_local_cone_pair(i, image, text, curv_d, metrics, out)


if __name__ == "__main__":
    main()
