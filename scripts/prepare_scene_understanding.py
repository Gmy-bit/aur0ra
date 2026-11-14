#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from tqdm import tqdm


ARO_IMAGE_ZIP_GDRIVE_ID = "1qaPlrwhGNMrR3a11iopZUT_GPP_LrgP9"
ARO_VGA_JSON_GDRIVE_ID = "13tWvOrNOLHxl3Rm9cR3geAdHx2qR3-Tw"
VL_CHECKLIST_RAW_BASE = (
    "https://raw.githubusercontent.com/om-ai-lab/VL-CheckList/main/data/Object"
)
VLCO_FILES = [
    "Size/vg_size/vg_obj_large.json",
    "Size/vg_size/vg_obj_medium.json",
    "Size/vg_size/vg_obj_small.json",
    "Size/vg_size/vg_subj_large.json",
    "Size/vg_size/vg_subj_medium.json",
    "Size/vg_size/vg_subj_small.json",
    "Location/vg_location/vg_obj_center.json",
    "Location/vg_location/vg_obj_mid.json",
    "Location/vg_location/vg_obj_margin.json",
    "Location/vg_location/vg_subj_center.json",
    "Location/vg_location/vg_subj_mid.json",
    "Location/vg_location/vg_subj_margin.json",
]
SUGARCREPE_RAW_BASE = "https://raw.githubusercontent.com/RAIVNLab/sugar-crepe/main/data"
SUGARCREPE_FILES = {
    "replace_obj": "replace_obj.json",
    "replace_att": "replace_att.json",
    "replace_rel": "replace_rel.json",
    "swap_obj": "swap_obj.json",
    "swap_att": "swap_att.json",
    "add_obj": "add_obj.json",
    "add_att": "add_att.json",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Table 5 scene-understanding datasets for SigMERU evaluation."
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/eval/scene_understanding",
        help="Output directory for prepared scene-understanding annotations and images.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Only convert existing downloaded raw files.",
    )
    parser.add_argument(
        "--prepare",
        choices=["all", "vga", "vlco", "sugarcrepe"],
        default="all",
        help="Which scene-understanding benchmark data to prepare.",
    )
    parser.add_argument(
        "--vlco-image-root",
        default=None,
        help="Directory containing VG_100K/ and VG_100K_2/ for VL-CO.",
    )
    parser.add_argument(
        "--sugarcrepe-image-root",
        default="coco/val2017",
        help=(
            "Directory containing COCO val2017 images for SugarCrepe. Relative paths "
            "are resolved under the eval data dir used by the evaluator."
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.prepare in ["all", "vga"]:
        try:
            prepare_vg_attribution(output_dir, skip_download=args.skip_download)
        except (
            FileNotFoundError,
            RuntimeError,
            subprocess.CalledProcessError,
            zipfile.BadZipFile,
        ) as exc:
            print("\nFailed to prepare VG-A automatically.", file=sys.stderr)
            print(str(exc), file=sys.stderr)
            print_vga_manual_download_instructions(output_dir)
            raise SystemExit(1) from exc

    if args.prepare in ["all", "vlco"]:
        vlco_image_root = (
            Path(args.vlco_image_root).expanduser().resolve()
            if args.vlco_image_root
            else output_dir / "vl_checklist_vg_images"
        )
        prepare_vl_checklist_object(
            output_dir=output_dir,
            image_root=vlco_image_root,
            skip_download=args.skip_download,
        )
    else:
        write_vlco_placeholder(output_dir)

    if args.prepare in ["all", "sugarcrepe"]:
        prepare_sugarcrepe(
            output_dir=output_dir,
            image_root=args.sugarcrepe_image_root,
            skip_download=args.skip_download,
        )

    print("\nScene-understanding data preparation status:")
    print(f"  VG-A annotation: {output_dir / 'vg_attribution.jsonl'}")
    print(f"  VG-A images:     {output_dir / 'images'}")
    print(f"  VL-CO annotation: {output_dir / 'vl_checklist_object.jsonl'}")
    print(f"  VL-CO images:     {output_dir / 'vl_checklist_vg_images'}")
    print(f"  SugarCrepe annotation: {output_dir / 'sugarcrepe.jsonl'}")


def prepare_vg_attribution(output_dir: Path, skip_download: bool) -> None:
    raw_dir = output_dir / "aro_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    image_zip = raw_dir / "vgr_vga_images.zip"
    raw_json = raw_dir / "visual_genome_attribution.json"

    if not skip_download:
        require_executable("gdown")
        if not image_zip.exists():
            run(
                [
                    "gdown",
                    "--no-cookies",
                    ARO_IMAGE_ZIP_GDRIVE_ID,
                    "--output",
                    str(image_zip),
                ]
            )
        if not raw_json.exists():
            run(
                [
                    "gdown",
                    "--id",
                    ARO_VGA_JSON_GDRIVE_ID,
                    "--output",
                    str(raw_json),
                ]
            )

    if not image_zip.exists():
        raise FileNotFoundError(
            f"Missing ARO image zip: {image_zip}. Run without --skip-download "
            "or download it manually."
        )
    if not raw_json.exists():
        raise FileNotFoundError(
            f"Missing ARO VG-A annotation: {raw_json}. Run without "
            "--skip-download or download it manually."
        )

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    marker = images_dir / ".vgr_vga_unzipped"
    if not marker.exists():
        try:
            with zipfile.ZipFile(image_zip, "r") as zf:
                zf.extractall(raw_dir)
        except zipfile.BadZipFile as exc:
            raise zipfile.BadZipFile(
                f"{image_zip} is not a valid zip file. It is likely an incomplete "
                "or interrupted Google Drive download. Delete it and re-download "
                "the ARO VG-R/VG-A image zip."
            ) from exc
        extracted_images = raw_dir / "images"
        if not extracted_images.exists():
            candidates = [p for p in raw_dir.rglob("images") if p.is_dir()]
            if not candidates:
                raise RuntimeError(
                    f"Could not find extracted ARO images directory under {raw_dir}"
                )
            extracted_images = candidates[0]
        for src in extracted_images.iterdir():
            dst = images_dir / src.name
            if src.is_file() and not dst.exists():
                shutil.copy2(src, dst)
        marker.write_text("ok\n")

    with raw_json.open("r") as f:
        rows = json.load(f)

    out_jsonl = output_dir / "vg_attribution.jsonl"
    with out_jsonl.open("w") as f:
        for row in tqdm(rows, desc="Converting VG-A"):
            converted = {
                "image": row["image_path"],
                "positive_caption": row["true_caption"],
                "negative_caption": row["false_caption"],
                "bbox_x": row["bbox_x"],
                "bbox_y": row["bbox_y"],
                "bbox_w": row["bbox_w"],
                "bbox_h": row["bbox_h"],
                "attributes": row.get("attributes"),
            }
            f.write(json.dumps(converted, ensure_ascii=False) + "\n")


def prepare_vl_checklist_object(
    output_dir: Path,
    image_root: Path,
    skip_download: bool,
) -> None:
    validate_vlco_image_root(image_root)
    raw_dir = output_dir / "vl_checklist_raw" / "Object"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not skip_download:
        for relative_path in VLCO_FILES:
            target = raw_dir / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                download_url(
                    f"{VL_CHECKLIST_RAW_BASE}/{relative_path}",
                    target,
                )

    missing = [str(raw_dir / path) for path in VLCO_FILES if not (raw_dir / path).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing VL-Checklist Object JSON files:\n  " + "\n  ".join(missing)
        )

    out_jsonl = output_dir / "vl_checklist_object.jsonl"
    written = 0
    skipped_missing_images = 0
    with out_jsonl.open("w") as out_f:
        for relative_path in VLCO_FILES:
            source_file = raw_dir / relative_path
            subset = source_file.stem
            with source_file.open("r") as f:
                rows = json.load(f)
            for row in tqdm(rows, desc=f"Converting VL-CO {subset}"):
                converted = convert_vlco_row(row, image_root, subset)
                if converted is None:
                    skipped_missing_images += 1
                    continue
                out_f.write(json.dumps(converted, ensure_ascii=False) + "\n")
                written += 1

    if written == 0:
        raise RuntimeError(f"No VL-CO samples were written to {out_jsonl}")
    if skipped_missing_images:
        print(
            f"Skipped {skipped_missing_images} VL-CO rows because images were missing.",
            file=sys.stderr,
        )


def validate_vlco_image_root(image_root: Path) -> None:
    missing_dirs = [name for name in ["VG_100K", "VG_100K_2"] if not (image_root / name).is_dir()]
    if missing_dirs:
        raise FileNotFoundError(
            f"VL-CO image root must contain VG_100K/ and VG_100K_2/. "
            f"Missing {missing_dirs} under {image_root}."
        )


def convert_vlco_row(
    row: list[Any],
    image_root: Path,
    subset: str,
) -> dict[str, Any] | None:
    image_rel, captions = row
    image_path = image_root / image_rel
    if not image_path.exists():
        return None

    positives = captions.get("POS") or []
    negatives = captions.get("NEG") or []
    if not positives or not negatives:
        raise ValueError(f"VL-CO row has no POS/NEG captions: {row!r}")

    return {
        "image": image_rel,
        "positive_caption": str(positives[0]).strip(),
        "negative_caption": str(negatives[0]).strip(),
        "subset": subset,
        "group": vlco_table3_group(subset),
    }


def vlco_table3_group(subset: str) -> str:
    for suffix in ["center", "mid", "margin", "large", "medium", "small"]:
        if subset.endswith(f"_{suffix}"):
            return suffix
    return subset


def prepare_sugarcrepe(output_dir: Path, image_root: str, skip_download: bool) -> None:
    raw_dir = output_dir / "sugarcrepe_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not skip_download:
        for filename in SUGARCREPE_FILES.values():
            target = raw_dir / filename
            if not target.exists():
                download_url(f"{SUGARCREPE_RAW_BASE}/{filename}", target)

    missing = [str(raw_dir / filename) for filename in SUGARCREPE_FILES.values() if not (raw_dir / filename).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing SugarCrepe JSON files:\n  " + "\n  ".join(missing)
        )

    out_jsonl = output_dir / "sugarcrepe.jsonl"
    written = 0
    with out_jsonl.open("w") as out_f:
        for group, filename in SUGARCREPE_FILES.items():
            source_file = raw_dir / filename
            with source_file.open("r") as f:
                rows = json.load(f)
            iterator = rows.values() if isinstance(rows, dict) else rows
            for row in tqdm(iterator, desc=f"Converting SugarCrepe {group}"):
                converted = convert_sugarcrepe_row(row, image_root, group)
                out_f.write(json.dumps(converted, ensure_ascii=False) + "\n")
                written += 1

    if written == 0:
        raise RuntimeError(f"No SugarCrepe samples were written to {out_jsonl}")


def convert_sugarcrepe_row(row: dict[str, Any], image_root: str, group: str) -> dict[str, Any]:
    filename = row.get("filename") or row.get("image") or row.get("image_path")
    if filename in [None, ""]:
        raise ValueError(f"SugarCrepe row has no filename/image field: {row!r}")
    caption = row.get("caption") or row.get("positive_caption")
    negative_caption = row.get("negative_caption")
    if caption in [None, ""] or negative_caption in [None, ""]:
        raise ValueError(f"SugarCrepe row has no caption pair: {row!r}")
    return {
        "image": str(Path(image_root) / str(filename)),
        "positive_caption": str(caption).strip(),
        "negative_caption": str(negative_caption).strip(),
        "group": group,
    }


def write_vlco_placeholder(output_dir: Path) -> None:
    readme = output_dir / "VL_CO_REQUIRED_FORMAT.txt"
    readme.write_text(
        "VL-CO is VL-Checklist-Object from Zhao et al. 2022.\n"
        "Prepare eval/scene_understanding/vl_checklist_object.jsonl "
        "with one JSON object per line.\n"
        "Supported minimal schema:\n"
        '{"image": "relative/or/absolute.jpg", "positive_caption": "correct", '
        '"negative_caption": "hard negative"}\n'
        "If image is relative, it is resolved under eval/scene_understanding/images by default.\n"
    )


def print_vga_manual_download_instructions(output_dir: Path) -> None:
    raw_dir = output_dir / "aro_raw"
    print(
        "\nManual VG-A preparation:\n"
        "1) Download the ARO VG-R/VG-A image zip from Google Drive:\n"
        f"   https://drive.google.com/uc?id={ARO_IMAGE_ZIP_GDRIVE_ID}\n"
        f"   Save as: {raw_dir / 'vgr_vga_images.zip'}\n"
        "2) Download the ARO VG-Attribution annotation from Google Drive:\n"
        f"   https://drive.google.com/uc?id={ARO_VGA_JSON_GDRIVE_ID}\n"
        f"   Save as: {raw_dir / 'visual_genome_attribution.json'}\n"
        "3) Re-run:\n"
        f"   python scripts/prepare_scene_understanding.py --output-dir {output_dir} --skip-download\n",
        file=sys.stderr,
    )


def require_executable(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(
            f"Required executable {name!r} was not found. Install it with: "
            "pip install gdown"
        )


def download_url(url: str, output_path: Path) -> None:
    print(f"Downloading {url} -> {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    last_error: Exception | None = None
    for attempt in range(1, 6):
        try:
            with urllib.request.urlopen(url, timeout=180) as response:
                tmp_path.write_bytes(response.read())
            tmp_path.replace(output_path)
            return
        except Exception as exc:
            last_error = exc
            if tmp_path.exists():
                tmp_path.unlink()
            print(f"  attempt {attempt}/5 failed: {exc}", file=sys.stderr)
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def run(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
