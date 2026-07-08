#!/usr/bin/env python3
"""Visualize saved DINO/Feature-3DGS feature maps as RGB PCA images."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from sklearn.decomposition import PCA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export RGB PCA visualizations for CxHxW feature maps.")
    parser.add_argument("--scene", default="dataset/20251204_scene_pinhole")
    parser.add_argument("--feature-dir", default="dinov3_dim128",
                        help="Feature directory relative to --scene, e.g. dinov3_dim128 or dinov3_fmap.")
    parser.add_argument("--images-dir", default="images",
                        help="Image directory relative to --scene. Used only for side-by-side sheets.")
    parser.add_argument("--output-dir", default="feature_vis",
                        help="Output directory relative to --scene.")
    parser.add_argument("--pattern", default="*.pth",
                        help="Feature glob pattern. Use '*_fmap_CxHxW.pt' for dinov3_fmap.")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--sample-step", type=int, default=3,
                        help="Pixel stride used to fit per-image PCA.")
    parser.add_argument("--max-pca-samples", type=int, default=200000,
                        help="Maximum pixels used to fit global PCA.")
    parser.add_argument("--global-pca", action="store_true",
                        help="Fit one PCA across selected maps instead of per-image PCA.")
    parser.add_argument("--vis-long-side", type=int, default=0,
                        help="Resize saved visualizations to this long side; 0 keeps feature-map size.")
    parser.add_argument("--no-sheet", action="store_true",
                        help="Do not create a contact sheet.")
    return parser.parse_args()


def load_feature(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ("feature", "features", "fmap", "semantic_feature"):
            if key in obj:
                obj = obj[key]
                break
        else:
            tensors = [v for v in obj.values() if torch.is_tensor(v)]
            if not tensors:
                raise ValueError(f"No tensor found in {path}")
            obj = tensors[0]
    if not torch.is_tensor(obj):
        raise TypeError(f"{path} did not contain a tensor")
    feat = obj.detach().float().cpu()
    if feat.ndim == 4 and feat.shape[0] == 1:
        feat = feat.squeeze(0)
    if feat.ndim != 3:
        raise ValueError(f"Expected CxHxW or HxWxC feature in {path}, got {tuple(feat.shape)}")
    # Saved maps in this project are CxHxW, e.g. 128x288x512. HxWxC maps
    # usually have the channel dimension last. DINOtxt patch maps are also
    # CxHxW, but C=1024 is larger than the spatial grid.
    if feat.shape[0] <= feat.shape[1] and feat.shape[0] <= feat.shape[2]:
        return feat.contiguous()
    if feat.shape[0] >= feat.shape[1] and feat.shape[0] >= feat.shape[2]:
        return feat.contiguous()
    if feat.shape[-1] <= feat.shape[0] and feat.shape[-1] <= feat.shape[1]:
        return feat.permute(2, 0, 1).contiguous()
    if feat.shape[-1] >= feat.shape[0] and feat.shape[-1] >= feat.shape[1]:
        return feat.permute(2, 0, 1).contiguous()
    raise ValueError(f"Cannot infer channel dimension for {path}: {tuple(feat.shape)}")


def sample_feature_pixels(feat: torch.Tensor, sample_step: int) -> np.ndarray:
    pixels = feat.permute(1, 2, 0).reshape(-1, feat.shape[0]).numpy()
    pixels = pixels[::max(1, sample_step)]
    finite = np.isfinite(pixels).all(axis=1)
    return pixels[finite]


def fit_pca(features: list[torch.Tensor], sample_step: int, max_samples: int) -> PCA:
    samples = []
    total = 0
    for feat in features:
        pixels = sample_feature_pixels(feat, sample_step)
        if len(pixels):
            if total + len(pixels) > max_samples:
                pixels = pixels[: max(0, max_samples - total)]
            samples.append(pixels)
            total += len(pixels)
        if total >= max_samples:
            break
    if not samples:
        raise RuntimeError("No finite feature samples for PCA")
    samples_np = np.concatenate(samples, axis=0)
    pca = PCA(n_components=3, random_state=42)
    pca.fit(samples_np)
    return pca


def fit_global_pca(paths: list[Path], sample_step: int, max_samples: int) -> PCA:
    samples = []
    total = 0
    for path in paths:
        pixels = sample_feature_pixels(load_feature(path), sample_step)
        if not len(pixels):
            continue
        remaining = max_samples - total
        if remaining <= 0:
            break
        if len(pixels) > remaining:
            idx = np.linspace(0, len(pixels) - 1, remaining, dtype=np.int64)
            pixels = pixels[idx]
        samples.append(pixels)
        total += len(pixels)
        print(f"[pca] sampled {total}/{max_samples} pixels")
        if total >= max_samples:
            break
    if not samples:
        raise RuntimeError("No finite feature samples for global PCA")
    pca = PCA(n_components=3, random_state=42)
    pca.fit(np.concatenate(samples, axis=0))
    return pca


def feature_to_rgb(feat: torch.Tensor, pca: PCA | None, sample_step: int) -> Image.Image:
    c, h, w = feat.shape
    pixels = feat.permute(1, 2, 0).reshape(-1, c).numpy()
    if pca is None:
        fit_pixels = pixels[::max(1, sample_step)]
        fit_pixels = fit_pixels[np.isfinite(fit_pixels).all(axis=1)]
        pca = PCA(n_components=3, random_state=42).fit(fit_pixels)
    rgb = pca.transform(np.nan_to_num(pixels, nan=0.0, posinf=0.0, neginf=0.0))
    lo = np.percentile(rgb, 1, axis=0)
    hi = np.percentile(rgb, 99, axis=0)
    rgb = (rgb - lo) / np.maximum(hi - lo, 1e-6)
    rgb = np.clip(rgb, 0.0, 1.0).reshape(h, w, 3)
    return Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")


def resize_vis(vis: Image.Image, long_side: int) -> Image.Image:
    if long_side <= 0:
        return vis
    w, h = vis.size
    scale = long_side / max(w, h)
    if abs(scale - 1.0) < 1e-6:
        return vis
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return vis.resize(new_size, Image.Resampling.NEAREST)


def feature_stem(path: Path) -> str:
    name = path.name
    for suffix in ("_dinov3_128.pth", "_fmap_CxHxW.pt", ".pth", ".pt"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in (".JPG", ".jpg", ".JPEG", ".jpeg", ".png", ".PNG"):
        path = images_dir / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def make_panel(stem: str, image_path: Path | None, vis: Image.Image) -> Image.Image:
    vis = vis.convert("RGB")
    label_h = 26
    if image_path and image_path.exists():
        src = Image.open(image_path).convert("RGB").resize(vis.size, Image.Resampling.BILINEAR)
        panel = Image.new("RGB", (vis.width * 2, vis.height + label_h), "white")
        panel.paste(src, (0, label_h))
        panel.paste(vis, (vis.width, label_h))
        labels = ("image", "feature PCA")
    else:
        panel = Image.new("RGB", (vis.width, vis.height + label_h), "white")
        panel.paste(vis, (0, label_h))
        labels = ("feature PCA",)
    draw = ImageDraw.Draw(panel)
    draw.text((6, 6), stem, fill=(0, 0, 0))
    if len(labels) == 2:
        draw.text((6, label_h - 18), labels[0], fill=(0, 0, 0))
        draw.text((vis.width + 6, label_h - 18), labels[1], fill=(0, 0, 0))
    return panel


def save_contact_sheet(panels: list[Image.Image], out_path: Path, columns: int = 3) -> None:
    if not panels:
        return
    w, h = panels[0].size
    rows = (len(panels) + columns - 1) // columns
    sheet = Image.new("RGB", (w * columns, h * rows), "white")
    for i, panel in enumerate(panels):
        sheet.paste(panel, ((i % columns) * w, (i // columns) * h))
    sheet.save(out_path)


def main() -> None:
    args = parse_args()
    scene = Path(args.scene)
    feature_dir = scene / args.feature_dir
    images_dir = scene / args.images_dir
    out_dir = scene / args.output_dir / args.feature_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(feature_dir.glob(args.pattern))[:: max(1, args.stride)]
    if args.limit > 0:
        paths = paths[: args.limit]
    if not paths:
        raise FileNotFoundError(f"No features found in {feature_dir} with pattern {args.pattern}")

    pca = fit_global_pca(paths, args.sample_step, args.max_pca_samples) if args.global_pca else None

    panels = []
    for path in paths:
        feat = load_feature(path)
        stem = feature_stem(path)
        vis = feature_to_rgb(feat, pca=pca, sample_step=args.sample_step)
        vis = resize_vis(vis, args.vis_long_side)
        out_path = out_dir / f"{stem}_feature_pca.png"
        vis.save(out_path)
        panels.append(make_panel(stem, find_image(images_dir, stem), vis))
        print(f"[save] {out_path} shape={tuple(feat.shape)}")

    if not args.no_sheet:
        sheet_path = out_dir / "contact_sheet.png"
        save_contact_sheet(panels, sheet_path)
        print(f"[save] {sheet_path}")


if __name__ == "__main__":
    main()
