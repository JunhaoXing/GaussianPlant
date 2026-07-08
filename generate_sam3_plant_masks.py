#!/usr/bin/env python3
"""Generate plant foreground masks for GaussianPlant scenes with SAM3.

The output is a binary grayscale mask per image, written as
<scene>/masks/<image_stem>.JPG by default. That filename convention matches the
current GaussianPlant COLMAP loader.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".PNG"}
DEFAULT_PROMPTS = ["plant"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use SAM3 text prompts to generate plant foreground masks."
    )
    parser.add_argument(
        "-s",
        "--scene",
        action="append",
        default=[],
        help="Scene folder containing images/. Can be passed multiple times.",
    )
    parser.add_argument(
        "--scenes-root",
        default=None,
        help="Optional root containing multiple scene folders.",
    )
    parser.add_argument(
        "--scene-glob",
        default="*",
        help="Glob used with --scenes-root. Only dirs with --image-dir are used.",
    )
    parser.add_argument("--image-dir", default="images", help="Image dir relative to each scene.")
    parser.add_argument("--mask-dir", default="masks", help="Mask dir relative to each scene.")
    parser.add_argument(
        "--mask-ext",
        default=".JPG",
        help="Mask extension. Default .JPG matches scene/dataset_readers.py.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="SAM3 text prompt. Repeat to union several prompts. Default: plant.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--sam3-resolution", type=int, default=1008)
    parser.add_argument("--checkpoint-path", default=None, help="Optional explicit sam3.pt path.")
    parser.add_argument(
        "--no-hf",
        action="store_true",
        help="Do not use HuggingFace cache/download. Requires --checkpoint-path.",
    )
    parser.add_argument(
        "--empty-policy",
        choices=("full", "empty", "skip"),
        default="full",
        help="What to write if SAM3 returns no mask. full is safest for training.",
    )
    parser.add_argument("--min-area", type=int, default=0, help="Drop SAM3 masks smaller than this.")
    parser.add_argument(
        "--max-area-frac",
        type=float,
        default=1.0,
        help="Drop individual masks larger than this image fraction before union.",
    )
    parser.add_argument("--morph-open", type=int, default=0, help="Open radius in pixels.")
    parser.add_argument("--morph-close", type=int, default=0, help="Close radius in pixels.")
    parser.add_argument("--dilate", type=int, default=0, help="Dilate radius in pixels.")
    parser.add_argument("--erode", type=int, default=0, help="Erode radius in pixels.")
    parser.add_argument("--fill-holes", action="store_true")
    parser.add_argument(
        "--min-component-area",
        type=int,
        default=0,
        help="Drop connected foreground components smaller than this many pixels.",
    )
    parser.add_argument(
        "--max-components",
        type=int,
        default=0,
        help="Keep only the N largest connected foreground components after union; 0 keeps all.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Debug: process first N images per scene.")
    parser.add_argument(
        "--save-overlays",
        action="store_true",
        help="Write quick QC overlays to <scene>/sam3_mask_vis/.",
    )
    parser.add_argument(
        "--save-detections",
        action="store_true",
        help="Write per-image prompt boxes/scores JSON to <scene>/sam3_mask_meta/.",
    )
    return parser.parse_args()


def lazy_imports():
    global Image, ImageOps, np, torch, cv2, build_sam3_image_model, Sam3Processor

    from PIL import Image, ImageOps
    import cv2
    import numpy as np
    import torch

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor


def list_images(image_dir: Path) -> list[Path]:
    paths = [p for p in image_dir.iterdir() if p.is_file() and p.suffix in IMAGE_EXTS]
    return sorted(paths, key=lambda p: p.name)


def resolve_scenes(args: argparse.Namespace) -> list[Path]:
    scenes = [Path(p).resolve() for p in args.scene]
    if args.scenes_root:
        root = Path(args.scenes_root).resolve()
        for path in sorted(root.glob(args.scene_glob)):
            if path.is_dir() and (path / args.image_dir).is_dir():
                scenes.append(path.resolve())

    deduped = []
    seen = set()
    for scene in scenes:
        if scene in seen:
            continue
        seen.add(scene)
        deduped.append(scene)
    if not deduped:
        raise SystemExit("Pass --scene or --scenes-root.")
    return deduped


def build_processor(args: argparse.Namespace):
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA is not available, but --device starts with cuda.")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    if args.no_hf and not args.checkpoint_path:
        raise ValueError("--no-hf requires --checkpoint-path")

    print("[SAM3] loading model")
    model = build_sam3_image_model(
        device=str(device),
        eval_mode=True,
        checkpoint_path=args.checkpoint_path,
        load_from_HF=not args.no_hf,
        enable_segmentation=True,
    )
    processor = Sam3Processor(
        model=model,
        resolution=args.sam3_resolution,
        device=str(device),
        confidence_threshold=args.confidence_threshold,
    )
    print("[SAM3] ready")
    return processor


def masks_from_state(state, min_area: int, max_area: int):
    if "masks" not in state or state["masks"] is None or state["masks"].numel() == 0:
        return None, []

    masks = state["masks"][:, 0].detach().to(torch.bool)
    boxes = state.get("boxes")
    scores = state.get("scores")
    keep = []
    records = []

    for idx in range(masks.shape[0]):
        area = int(masks[idx].sum().item())
        score = float(scores[idx].detach().cpu().item()) if scores is not None else 0.0
        box = boxes[idx].detach().cpu().tolist() if boxes is not None else None
        kept = area >= min_area and area <= max_area
        records.append(
            {
                "idx": int(idx),
                "area": area,
                "score": score,
                "box_xyxy": [float(v) for v in box] if box is not None else None,
                "kept": bool(kept),
            }
        )
        if kept:
            keep.append(idx)

    if not keep:
        return None, records
    union = masks[keep].any(dim=0).detach().cpu().numpy()
    return union, records


def fill_binary_holes(mask):
    h, w = mask.shape
    flood = np.pad(mask.astype("uint8") * 255, 1, mode="constant", constant_values=0)
    canvas = flood.copy()
    ff_mask = np.zeros((h + 4, w + 4), dtype=np.uint8)
    cv2.floodFill(canvas, ff_mask, (0, 0), 255)
    canvas = canvas[1 : h + 1, 1 : w + 1]
    holes = canvas == 0
    return mask | holes


def morph_mask(mask, args: argparse.Namespace):
    out = mask.astype("uint8")

    def kernel(radius: int):
        size = radius * 2 + 1
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    if args.morph_open > 0:
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel(args.morph_open))
    if args.morph_close > 0:
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel(args.morph_close))
    if args.dilate > 0:
        out = cv2.dilate(out, kernel(args.dilate))
    if args.erode > 0:
        out = cv2.erode(out, kernel(args.erode))
    out = out.astype(bool)
    if args.fill_holes:
        out = fill_binary_holes(out)
    out = filter_components(out, args.min_component_area, args.max_components)
    return out


def filter_components(mask, min_area: int = 0, max_components: int = 0):
    if not mask.any() or (min_area <= 0 and max_components <= 0):
        return mask

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype("uint8"), connectivity=8
    )
    components = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            components.append((label, area))
    if max_components > 0:
        components = sorted(components, key=lambda item: item[1], reverse=True)[:max_components]
    if not components:
        return np.zeros_like(mask, dtype=bool)

    keep_labels = np.array([label for label, _area in components], dtype=labels.dtype)
    return np.isin(labels, keep_labels)


def generate_mask(processor, image, prompts: Iterable[str], args: argparse.Namespace):
    width, height = image.size
    max_area = int(width * height * max(0.0, args.max_area_frac))
    base_state = processor.set_image(image)
    union = np.zeros((height, width), dtype=bool)
    prompt_records = []

    for prompt in prompts:
        state = processor.set_text_prompt(prompt=prompt, state=base_state)
        prompt_mask, records = masks_from_state(state, args.min_area, max_area)
        if prompt_mask is not None:
            union |= prompt_mask
        prompt_records.append({"prompt": prompt, "detections": records})

    return morph_mask(union, args), prompt_records


def save_mask(mask, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr = (mask.astype("uint8") * 255)
    image = Image.fromarray(arr, mode="L")
    if output_path.suffix.lower() in {".jpg", ".jpeg"}:
        image.save(output_path, quality=95)
    else:
        image.save(output_path)


def save_overlay(image, mask, output_path: Path, alpha: float = 0.45):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    color = np.array([40, 220, 80], dtype=np.float32)
    out = rgb.copy()
    out[mask] = rgb[mask] * (1.0 - alpha) + color * alpha
    Image.fromarray(np.clip(out, 0, 255).astype("uint8")).save(output_path, quality=92)


def process_scene(scene: Path, processor, args: argparse.Namespace) -> dict:
    image_dir = scene / args.image_dir
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image dir not found: {image_dir}")
    mask_dir = scene / args.mask_dir
    vis_dir = scene / "sam3_mask_vis"
    meta_dir = scene / "sam3_mask_meta"
    prompts = args.prompt or DEFAULT_PROMPTS

    image_paths = list_images(image_dir)
    if args.limit > 0:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    summary_path = scene / "sam3_mask_summary.jsonl"
    summary = {"scene": str(scene), "images": len(image_paths), "saved": 0, "skipped": 0, "empty": 0}
    print(f"[scene] {scene} images={len(image_paths)} prompts={prompts}")
    with summary_path.open("w", encoding="utf-8") as summary_file:
        for idx, image_path in enumerate(image_paths, 1):
            out_path = mask_dir / f"{image_path.stem}{args.mask_ext}"
            if out_path.exists() and not args.overwrite:
                print(f"[skip] {idx}/{len(image_paths)} {image_path.name}")
                summary["skipped"] += 1
                continue

            image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
            mask, prompt_records = generate_mask(processor, image, prompts, args)
            detected_pixels = int(mask.sum())
            empty_policy = None
            if detected_pixels == 0:
                summary["empty"] += 1
                empty_policy = args.empty_policy
                if args.empty_policy == "skip":
                    print(f"[empty:skip] {idx}/{len(image_paths)} {image_path.name}")
                    summary["skipped"] += 1
                    continue
                if args.empty_policy == "full":
                    mask = np.ones((image.height, image.width), dtype=bool)

            save_mask(mask, out_path)
            if args.save_overlays:
                save_overlay(image, mask, vis_dir / f"{image_path.stem}.jpg")
            if args.save_detections:
                meta_dir.mkdir(parents=True, exist_ok=True)
                with (meta_dir / f"{image_path.stem}.json").open("w", encoding="utf-8") as f:
                    json.dump(prompt_records, f, indent=2)

            ratio = float(mask.mean())
            row = {
                "image": image_path.name,
                "mask": str(out_path.relative_to(scene)),
                "width": image.width,
                "height": image.height,
                "foreground_pixels": int(mask.sum()),
                "foreground_ratio": ratio,
                "empty_policy": empty_policy,
            }
            summary_file.write(json.dumps(row) + "\n")
            summary["saved"] += 1
            print(
                f"[save] {idx}/{len(image_paths)} {out_path.name} "
                f"fg={ratio:.3f}"
            )

    print(
        f"[done] {scene.name}: saved={summary['saved']} "
        f"skipped={summary['skipped']} empty={summary['empty']} "
        f"summary={summary_path}"
    )
    return summary


def main():
    args = parse_args()
    if args.mask_ext and not args.mask_ext.startswith("."):
        args.mask_ext = f".{args.mask_ext}"
    lazy_imports()
    scenes = resolve_scenes(args)
    processor = build_processor(args)
    totals = []
    for scene in scenes:
        totals.append(process_scene(scene, processor, args))
    print("[all done]")
    for item in totals:
        print(
            f"  {Path(item['scene']).name}: saved={item['saved']} "
            f"skipped={item['skipped']} empty={item['empty']}"
        )


if __name__ == "__main__":
    main()
