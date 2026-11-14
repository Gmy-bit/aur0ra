#!/usr/bin/env python
"""Build a small, clean reference set from WebDataset TAR shards.

Outputs:
  output_dir/
    images/000000.jpg
    metadata.jsonl
    reference_set_summary.json

The script is intentionally independent of the training loader so broken images,
empty captions, missing fields, and partially corrupted TAR members can be skipped
without crashing the run.
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

_IMAGE_EXTS = {"jpg", "jpeg", "png", "webp"}
_TEXT_EXTS = {"txt", "text", "caption", "json"}


def _split_key_field(name: str) -> tuple[str, str]:
    base = Path(name).name
    if "." not in base:
        return base, ""
    key, field = base.split(".", 1)
    return key, field.lower()


def _field_ext(field: str) -> str:
    return field.rsplit(".", 1)[-1].lower()


def _is_box_field(field: str) -> bool:
    f = field.lower()
    return any(tok in f for tok in ("box", "bbox", "region", "parent"))


def _is_image_field(field: str) -> bool:
    return _field_ext(field) in _IMAGE_EXTS


def _is_text_field(field: str) -> bool:
    return _field_ext(field) in _TEXT_EXTS


def _read_member(tar: tarfile.TarFile, member: tarfile.TarInfo) -> bytes | None:
    try:
        f = tar.extractfile(member)
        if f is None:
            return None
        return f.read()
    except Exception:
        return None


def _decode_image(data: bytes, image_size: int | None = None) -> Image.Image | None:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        img = img.convert("RGB")
        if image_size and image_size > 0:
            img.thumbnail((image_size, image_size), Image.Resampling.BICUBIC)
        return img
    except Exception:
        return None


def _decode_text(data: bytes) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin1"):
        try:
            text = data.decode(enc).strip()
            break
        except Exception:
            text = ""
    if not text:
        return ""
    # Some WebDataset pipelines store JSON records; try common caption keys.
    if text.startswith("{"):
        try:
            obj = json.loads(text)
            for key in ("caption", "text", "txt", "description"):
                if key in obj and str(obj[key]).strip():
                    return str(obj[key]).strip()
        except Exception:
            pass
    return text


def _choose_fields(sample: dict[str, bytes]):
    image_candidates: list[tuple[str, bytes]] = []
    text_candidates: list[tuple[str, str]] = []
    box_image_candidates: list[tuple[str, bytes]] = []
    box_text_candidates: list[tuple[str, str]] = []
    raw_boxes: list[dict] = []

    for field, data in sample.items():
        if _is_image_field(field):
            (box_image_candidates if _is_box_field(field) else image_candidates).append((field, data))
        elif _is_text_field(field):
            text = _decode_text(data)
            (box_text_candidates if _is_box_field(field) else text_candidates).append((field, text))
        elif "json" in field.lower() or "bbox" in field.lower():
            try:
                raw_boxes.append(json.loads(data.decode("utf-8")))
            except Exception:
                pass

    def pick_image(items):
        items = sorted(items, key=lambda x: ("jpg" not in x[0], "png" not in x[0], len(x[1]), x[0]))
        return items[0] if items else None

    def pick_text(items):
        items = sorted(items, key=lambda x: (len(x[1]) == 0, "txt" not in x[0], x[0]))
        return items[0] if items else None

    return pick_image(image_candidates), pick_text(text_candidates), pick_image(box_image_candidates), pick_text(box_text_candidates), raw_boxes


def iter_grouped_samples(tar_paths: Iterable[str], rng: random.Random):
    for tar_path in tar_paths:
        try:
            with tarfile.open(tar_path, "r:*") as tar:
                grouped: dict[str, dict[str, bytes]] = defaultdict(dict)
                members = [m for m in tar.getmembers() if m.isfile()]
                rng.shuffle(members)
                for m in members:
                    key, field = _split_key_field(m.name)
                    if not key or not field:
                        continue
                    data = _read_member(tar, m)
                    if data is None:
                        continue
                    grouped[key][field] = data
                keys = list(grouped.keys())
                rng.shuffle(keys)
                for k in keys:
                    yield tar_path, k, grouped[k]
        except Exception as e:
            print(f"[WARN] failed to read shard {tar_path}: {e}")
            continue


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--wds_shards", required=True, help="Glob for WebDataset TAR shards, e.g. '/path/*.tar'.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_samples", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image_size", type=int, default=224, help="Max side used when saving reference JPEGs. 0 keeps original size.")
    p.add_argument("--no_box", action="store_true", help="Do not save box image/text metadata even if present.")
    p.add_argument("--max_shards", type=int, default=0, help="0 means scan all matched shards until enough samples are found.")
    args = p.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.output_dir)
    image_dir = out / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    shards = sorted(glob.glob(args.wds_shards))
    if not shards:
        raise FileNotFoundError(f"No shards match {args.wds_shards!r}")
    rng.shuffle(shards)
    if args.max_shards > 0:
        shards = shards[: args.max_shards]

    counts = dict(seen=0, saved=0, skipped_missing=0, skipped_empty_caption=0, skipped_bad_image=0, with_box=0)
    meta_path = out / "metadata.jsonl"
    with meta_path.open("w", encoding="utf-8") as fw:
        for shard, key, sample in iter_grouped_samples(shards, rng):
            if counts["saved"] >= args.num_samples:
                break
            counts["seen"] += 1
            image_item, text_item, box_image_item, box_text_item, raw_boxes = _choose_fields(sample)
            if image_item is None or text_item is None:
                counts["skipped_missing"] += 1
                continue
            image_field, image_bytes = image_item
            text_field, caption = text_item
            if not caption:
                counts["skipped_empty_caption"] += 1
                continue
            image = _decode_image(image_bytes, image_size=args.image_size)
            if image is None:
                counts["skipped_bad_image"] += 1
                continue

            idx = counts["saved"]
            rel = f"images/{idx:06d}.jpg"
            image.save(out / rel, quality=95)
            boxes: list[dict] = []
            if not args.no_box:
                if box_text_item is not None and box_text_item[1]:
                    boxes.append({"caption": box_text_item[1], "text_field": box_text_item[0]})
                if box_image_item is not None:
                    boxes.append({"image_field": box_image_item[0], "has_box_image": True})
                for rb in raw_boxes:
                    boxes.append({"raw": rb})
            if boxes:
                counts["with_box"] += 1

            rec = {
                "id": f"{idx:06d}",
                "image_path": rel,
                "caption": caption,
                "boxes": boxes,
                "source_shard": shard,
                "source_key": key,
                "image_field": image_field,
                "text_field": text_field,
            }
            fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
            counts["saved"] += 1
            if counts["saved"] % 50 == 0:
                print(f"[INFO] saved {counts['saved']}/{args.num_samples}")

    summary = {
        **counts,
        "box_available_ratio": counts["with_box"] / max(1, counts["saved"]),
        "output_dir": str(out),
        "metadata": str(meta_path),
        "num_shards_scanned_limit": len(shards),
    }
    (out / "reference_set_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
