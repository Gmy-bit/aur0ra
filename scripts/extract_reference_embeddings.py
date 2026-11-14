#!/usr/bin/env python
"""Extract Lorentz embeddings and geometry diagnostics for a reference set."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sigmeru import lorentz as L
from sigmeru.config import LazyConfig, LazyFactory
from sigmeru.models import (
    compute_exterior_angle,
    compute_exterior_angle_cos,
    pairwise_lorentz_distance,
    pairwise_rapidity,
    pairwise_root_angle,
    pairwise_root_angle_cos,
    radius_from_origin,
)
from sigmeru.tokenizer import Tokenizer
from sigmeru.utils.checkpointing import CheckpointManager


class ReferenceDataset(Dataset):
    def __init__(self, reference_dir: str | Path, image_size: int = 224, max_samples: int = 0):
        self.root = Path(reference_dir)
        meta = self.root / "metadata.jsonl"
        if not meta.exists():
            raise FileNotFoundError(f"metadata.jsonl not found: {meta}")
        self.records = [json.loads(line) for line in meta.read_text(encoding="utf-8").splitlines() if line.strip()]
        if max_samples and max_samples > 0:
            self.records = self.records[:max_samples]
        self.transform = T.Compose([
            T.Resize(image_size, T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        r = self.records[idx]
        img = Image.open(self.root / r["image_path"]).convert("RGB")
        return {
            "index": idx,
            "image": self.transform(img),
            "caption": r.get("caption", ""),
            "image_path": r.get("image_path", ""),
            "boxes": r.get("boxes", []),
        }


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "index": torch.tensor([b["index"] for b in batch], dtype=torch.long),
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "caption": [b["caption"] for b in batch],
        "image_path": [b["image_path"] for b in batch],
        "boxes": [b["boxes"] for b in batch],
    }


def load_model(train_config: str, checkpoint: str, device: torch.device):
    cfg = LazyConfig.load(train_config)
    model = LazyFactory.build_model(cfg, device).to(device).eval()
    CheckpointManager(Path(checkpoint).parent, model=model).load(checkpoint)
    return model


@torch.inference_mode()
def encode(model, dataset: ReferenceDataset, batch_size: int, num_workers: int, device: torch.device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    tokenizer = Tokenizer()
    text_l, image_l, captions, paths, boxes = [], [], [], [], []
    for batch in tqdm(loader, desc="Extracting reference embeddings"):
        images = batch["image"].to(device, non_blocking=True)
        tokens = tokenizer(batch["caption"])
        image_feats = model.encode_image(images, project=True)
        text_feats = model.encode_text(tokens, project=True)
        image_l.append(image_feats.detach().cpu())
        text_l.append(text_feats.detach().cpu())
        captions.extend(batch["caption"])
        paths.extend(batch["image_path"])
        boxes.extend(batch["boxes"])
    return torch.cat(text_l, dim=0), torch.cat(image_l, dim=0), captions, paths, boxes


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reference_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--train_config", required=True)
    p.add_argument("--output_dir", default="")
    p.add_argument("--output_path", default="")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--max_samples", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir or (Path(args.output_path).parent if args.output_path else "reference_outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_path = Path(args.output_path) if args.output_path else out_dir / "reference_embeddings.pt"

    dataset = ReferenceDataset(args.reference_dir, image_size=args.image_size, max_samples=args.max_samples)
    model = load_model(args.train_config, args.checkpoint, device)
    curv = model.curv.exp().detach().cpu()
    text_l, image_l, captions, paths, boxes = encode(model, dataset, args.batch_size, args.num_workers, device)

    text = text_l.to(device)
    image = image_l.to(device)
    curv_d = curv.to(device)
    with torch.inference_mode():
        text_space_norm = torch.norm(text, dim=-1).cpu()
        image_space_norm = torch.norm(image, dim=-1).cpu()
        text_radius = radius_from_origin(text, curv_d).cpu()
        image_radius = radius_from_origin(image, curv_d).cpu()
        dist_mat = pairwise_lorentz_distance(text, image, curv_d).cpu()
        rapidity_mat = pairwise_rapidity(text, image, curv_d).cpu()
        root_cos_mat = pairwise_root_angle_cos(text, image, curv_d).cpu()
        root_angle_mat = pairwise_root_angle(text, image, curv_d).cpu()
        pos_dist = dist_mat.diag()
        pos_rapid = rapidity_mat.diag()
        pos_root_cos = root_cos_mat.diag()
        pos_root_angle = root_angle_mat.diag()
        cos_alpha_pair = compute_exterior_angle_cos(text, image, curv_d).cpu()
        alpha_pair = torch.acos(cos_alpha_pair.clamp(-1 + 1e-8, 1 - 1e-8))
        pos_cos_alpha = cos_alpha_pair.diag()
        pos_alpha = alpha_pair.diag()
        aperture = L.half_aperture(text, curv_d).cpu()
        ext_minus = pos_alpha - aperture
        inside = ext_minus <= 0

    data = {
        "text_lorentz": text_l,
        "image_lorentz": image_l,
        "text_radius": text_radius,
        "image_radius": image_radius,
        "text_space_norm": text_space_norm,
        "image_space_norm": image_space_norm,
        "pairwise_lorentz_distance": dist_mat,
        "pairwise_rapidity": rapidity_mat,
        "pairwise_root_angle_cos": root_cos_mat,
        "pairwise_root_angle": root_angle_mat,
        "positive_root_angle": pos_root_angle,
        "positive_root_angle_cos": pos_root_cos,
        "positive_alpha": pos_alpha,
        "positive_cos_alpha": pos_cos_alpha,
        "positive_aperture": aperture,
        "positive_inside_cone": inside,
        "positive_ext_minus_aper": ext_minus,
        "curv": float(curv.item()),
        "captions": captions,
        "image_paths": paths,
        "boxes": boxes,
    }
    torch.save(data, emb_path)

    rows = []
    for i in range(len(captions)):
        rows.append({
            "index": i,
            "caption": captions[i],
            "image_path": paths[i],
            "text_radius": float(text_radius[i]),
            "image_radius": float(image_radius[i]),
            "text_space_norm": float(text_space_norm[i]),
            "image_space_norm": float(image_space_norm[i]),
            "positive_distance": float(pos_dist[i]),
            "positive_rapidity": float(pos_rapid[i]),
            "positive_root_angle": float(pos_root_angle[i]),
            "positive_root_angle_cos": float(pos_root_cos[i]),
            "positive_alpha": float(pos_alpha[i]),
            "positive_cos_alpha": float(pos_cos_alpha[i]),
            "positive_aperture": float(aperture[i]),
            "positive_inside_cone": bool(inside[i]),
            "positive_ext_minus_aper": float(ext_minus[i]),
        })
    write_csv(out_dir / "reference_pair_metrics.csv", rows)
    metrics = {
        "num_samples": len(captions),
        "curv": float(curv.item()),
        "positive_inside_cone_ratio": float(inside.float().mean()),
        "positive_distance_mean": float(pos_dist.mean()),
        "positive_rapidity_mean": float(pos_rapid.mean()),
        "positive_root_angle_mean": float(pos_root_angle.mean()),
        "positive_root_angle_cos_mean": float(pos_root_cos.mean()),
        "positive_alpha_mean": float(pos_alpha.mean()),
        "positive_cos_alpha_mean": float(pos_cos_alpha.mean()),
        "text_radius_mean": float(text_radius.mean()),
        "image_radius_mean": float(image_radius.mean()),
    }
    (out_dir / "reference_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"saved": str(emb_path), **metrics}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
