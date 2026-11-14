#!/usr/bin/env python
"""Create a small, clean image-text reference set from WebDataset TAR files.

The script intentionally avoids WebDataset's iterator so it can skip broken
samples gracefully and preserve enough metadata for visualization/debugging.

Outputs:
  output_dir/
    metadata.jsonl
    images/000000.jpg
    boxes/000000_box.jpg          # optional, if available and --include-boxes
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import random
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from PIL import Image

# -----------------------------------------------------------------------------
# Editable defaults for quick use on your server.
# -----------------------------------------------------------------------------
DEFAULT_TAR_GLOB = "datasets/train/GRIT/processed/*.tar"
DEFAULT_OUTPUT_DIR = "reference_sets/grit_ref500"
DEFAULT_NUM_SAMPLES = 500
DEFAULT_INCLUDE_BOXES = True
DEFAULT_SEED = 0
DEFAULT_MAX_TARS = 0  # 0 = scan all TARs until enough samples are collected.

_IMAGE_EXTS = {"jpg", "jpeg", "png", "webp"}
_TEXT_EXTS = {"txt", "text", "caption"}


def _sample_key_and_field(name: str) -> tuple[str, str]:
    base = Path(name).name
    if "." not in base:
        return base, ""
    key, field = base.split(".", 1)
    return key, field.lower()


def _is_box_field(field: str) -> bool:
    f = field.lower()
    return any(token in f for token in ("box", "parent", "bbox", "region"))


def _is_image_field(field: str) -> bool:
    return field.split(".")[-1].lower() in _IMAGE_EXTS


def _is_text_field(field: str) -> bool:
    return field.split(".")[-1].lower() in _TEXT_EXTS


def _read_member(tar: tarfile.TarFile, member: tarfile.TarInfo) -> bytes | None:
    try:
        f = tar.extractfile(member)
        if f is None:
            return None
        return f.read()
    except Exception:
        return None


def _decode_image(data: bytes) -> Image.Image | None:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img.convert("RGB")
    except Exception:
        return None


def _decode_text(data: bytes) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin1"):
        try:
            return data.decode(enc).strip()
        except Exception:
            continue
    return ""


def _choose_fields(sample: dict[str, bytes]) -> tuple[tuple[str, bytes] | None, tuple[str, str] | None, tuple[str, bytes] | None, tuple[str, str] | None]:
    image_candidates = []
    text_candidates = []
    box_image_candidates = []
    box_text_candidates = []

    for field, data in sample.items():
        if _is_image_field(field):
            if _is_box_field(field):
                box_image_candidates.append((field, data))
            else:
                image_candidates.append((field, data))
        elif _is_text_field(field):
            text = _decode_text(data)
            if _is_box_field(field):
                box_text_candidates.append((field, text))
            else:
                text_candidates.append((field, text))

    def pref_image(items):
        # Prefer child/full image fields over accidental thumbnails.
        items = sorted(items, key=lambda x: ("child" not in x[0], "jpg" not in x[0], x[0]))
        return items[0] if items else None

    def pref_text(items):
        items = sorted(items, key=lambda x: ("child" not in x[0], len(x[1]) == 0, x[0]))
        return items[0] if items else None

    return pref_image(image_candidates), pref_text(text_candidates), pref_image(box_image_candidates), pref_text(box_text_candidates)


def iter_samples_from_tars(tar_paths: Iterable[str], shuffle_members: bool = False):
    for tar_path in tar_paths:
        try:
            with tarfile.open(tar_path, "r:*") as tar:
                members = [m for m in tar.getmembers() if m.isfile()]
                if shuffle_members:
                    random.shuffle(members)
                grouped: dict[str, dict[str, bytes]] = defaultdict(dict)
                for m in members:
                    key, field = _sample_key_and_field(m.name)
                    if not key or not field:
                        continue
                    if not (_is_image_field(field) or _is_text_field(field)):
                        continue
                    data = _read_member(tar, m)
                    if data is None:
                        continue
                    grouped[key][field] = data

                keys = list(grouped.keys())
                if shuffle_members:
                    random.shuffle(keys)
                for key in keys:
                    yield tar_path, key, grouped[key]
        except Exception as e:
            print(f"[WARN] failed to read {tar_path}: {e}")
            continue


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tarfiles", default=DEFAULT_TAR_GLOB)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--max-tars", type=int, default=DEFAULT_MAX_TARS)
    p.add_argument("--include-boxes", action=argparse.BooleanOptionalAction, default=DEFAULT_INCLUDE_BOXES)
    p.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    random.seed(args.seed)
    out = Path(args.output_dir)
    image_dir = out / "images"
    box_dir = out / "boxes"
    image_dir.mkdir(parents=True, exist_ok=True)
    if args.include_boxes:
        box_dir.mkdir(parents=True, exist_ok=True)

    tar_paths = sorted(glob.glob(args.tarfiles))
    if args.shuffle:
        random.shuffle(tar_paths)
    if args.max_tars and args.max_tars > 0:
        tar_paths = tar_paths[: args.max_tars]
    if not tar_paths:
        raise FileNotFoundError(f"No TAR files match: {args.tarfiles}")

    meta_path = out / "metadata.jsonl"
    n_ok = n_seen = n_bad = 0
    with meta_path.open("w", encoding="utf-8") as fw:
        for tar_path, sample_key, sample in iter_samples_from_tars(tar_paths, args.shuffle):
            if n_ok >= args.num_samples:
                break
            n_seen += 1
            image_item, text_item, box_image_item, box_text_item = _choose_fields(sample)
            if image_item is None or text_item is None:
                n_bad += 1
                continue
            image_field, image_data = image_item
            text_field, caption = text_item
            if not caption:
                n_bad += 1
                continue
            img = _decode_image(image_data)
            if img is None:
                n_bad += 1
                continue

            idx = n_ok
            image_rel = f"images/{idx:06d}.jpg"
            img.save(out / image_rel, quality=95)

            record = {
                "id": idx,
                "source_tar": tar_path,
                "sample_key": sample_key,
                "image": image_rel,
                "caption": caption,
                "image_field": image_field,
                "text_field": text_field,
                "has_box": False,
            }

            if args.include_boxes and box_image_item is not None:
                box_img = _decode_image(box_image_item[1])
                if box_img is not None:
                    box_rel = f"boxes/{idx:06d}_box.jpg"
                    box_img.save(out / box_rel, quality=95)
                    record["box_image"] = box_rel
                    record["box_image_field"] = box_image_item[0]
                    record["has_box"] = True
                    if box_text_item is not None and box_text_item[1]:
                        record["box_caption"] = box_text_item[1]
                        record["box_text_field"] = box_text_item[0]

            fw.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_ok += 1
            if n_ok % 50 == 0:
                print(f"[INFO] collected {n_ok}/{args.num_samples} valid samples")

    print(f"[DONE] saved {n_ok} samples to {out}")
    print(f"[DONE] seen={n_seen}, skipped_bad_or_incomplete={n_bad}, metadata={meta_path}")


if __name__ == "__main__":
    main()
