#!/usr/bin/env python3
"""Prepare resized image folders for COLMAP scenes.

Default behavior center-crops images to the target aspect ratio, then resizes
to the exact target size. This avoids geometric stretching when the source
aspect ratio differs from 1920x1080.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG"}


def parse_args():
    parser = argparse.ArgumentParser(description="Resize raw multi-view images into scene/images.")
    parser.add_argument(
        "--src",
        default="dataset/20251204_multi",
        help="Directory containing raw input images.",
    )
    parser.add_argument(
        "--dst",
        default="dataset/20251204_scene/images",
        help="Output images directory for the COLMAP scene.",
    )
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument(
        "--mode",
        choices=("crop", "fit", "stretch"),
        default="crop",
        help=(
            "crop: center-crop to target aspect then exact resize; "
            "fit: preserve aspect inside target bounds; "
            "stretch: exact resize with distortion."
        ),
    )
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def list_images(src: Path):
    return sorted(p for p in src.iterdir() if p.is_file() and p.suffix in IMAGE_EXTS)


def center_crop_to_aspect(image: Image.Image, target_aspect: float) -> Image.Image:
    width, height = image.size
    aspect = width / height
    if abs(aspect - target_aspect) < 1e-6:
        return image

    if aspect > target_aspect:
        new_width = round(height * target_aspect)
        left = (width - new_width) // 2
        return image.crop((left, 0, left + new_width, height))

    new_height = round(width / target_aspect)
    top = (height - new_height) // 2
    return image.crop((0, top, width, top + new_height))


def resize_image(image: Image.Image, width: int, height: int, mode: str) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    if mode == "crop":
        image = center_crop_to_aspect(image, width / height)
        return image.resize((width, height), Image.Resampling.LANCZOS)
    if mode == "fit":
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        return image.copy()
    return image.resize((width, height), Image.Resampling.LANCZOS)


def main():
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)

    if not src.is_dir():
        raise FileNotFoundError(f"Input directory not found: {src}")

    images = list_images(src)
    if not images:
        raise FileNotFoundError(f"No images found in: {src}")

    dst.mkdir(parents=True, exist_ok=True)
    print(f"[prepare] source: {src.resolve()}")
    print(f"[prepare] output: {dst.resolve()}")
    print(f"[prepare] images: {len(images)}")
    print(f"[prepare] mode: {args.mode}, target: {args.width}x{args.height}")

    skipped = 0
    for path in tqdm(images, desc="Resizing"):
        out_path = dst / path.name
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        with Image.open(path) as image:
            resized = resize_image(image, args.width, args.height, args.mode)
            resized.save(out_path, quality=args.quality, optimize=True)

    print(f"[prepare] done. skipped existing: {skipped}")


if __name__ == "__main__":
    main()
