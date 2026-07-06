#!/usr/bin/env python3
"""Extract DINOv3 dense feature maps for GaussianPlant Step 1a.

The script follows the DINOv3 README path:

    torch.hub.load(<local dinov3 repo>, <model>, source="local", weights=...)
    model.get_intermediate_layers(..., reshape=True, norm=True)

It then upsamples patch features, fits/applies PCA to 128 dimensions, and writes
per-view feature maps consumable by GaussianPlant and Feature-3DGS.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".PNG"}
DINO_TO_JAFAR_CKPT = {
    "dinov3_vits16": "vit_small_patch16_dinov3.lvd1689m.pth",
    "dinov3_vits16plus": "vit_small_plus_patch16_dinov3.lvd1689m.pth",
    "dinov3_vitb16": "vit_base_patch16_dinov3.lvd1689m.pth",
    "dinov3_vitl16": "vit_large_patch16_dinov3.lvd1689m.pth",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Export DINOv3 dense PCA-128 feature maps.")
    parser.add_argument("-s", "--scene", required=True, help="Scene folder containing images/")
    parser.add_argument("--image-dir", default="images", help="Image directory relative to scene")
    parser.add_argument("--output-dir", default="dinov3_dim128",
                        help="GaussianPlant feature dir relative to scene")
    parser.add_argument("--feature3dgs-dir", default="dinov3_fmap",
                        help="Optional Feature-3DGS-compatible dir; set empty to disable")
    parser.add_argument("--dinov3-repo", default="third_party/dinov3")
    parser.add_argument("--dinov3-model", default="dinov3_vitl16")
    parser.add_argument("--dinov3-weights", default=None,
                        help="DINOv3 checkpoint path/URL. If omitted, hub default is used.")
    parser.add_argument("--checkpoint-dir", default="checkpoints",
                        help="Directory used to auto-find JAFAR/DINOv3 checkpoints")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16",
                        help="DINOv3 inference dtype. float16 can produce NaNs on some GPUs.")
    parser.add_argument("--out-dim", type=int, default=128)
    parser.add_argument("--pca-path", default=None,
                        help="Path to save/load PCA. Default: <scene_parent>/dinov3_pca.pth")
    parser.add_argument("--fit-pca", action="store_true",
                        help="Fit PCA even if --pca-path already exists")
    parser.add_argument("--samples-per-image", type=int, default=4096)
    parser.add_argument("--max-pca-samples", type=int, default=200000)
    parser.add_argument("--transform-chunk", type=int, default=262144,
                        help="Pixels per PCA transform chunk")
    parser.add_argument("--max-long-side", type=int, default=0,
                        help="Resize image long side before DINO/JAFAR; 0 keeps original size")
    parser.add_argument("--output-long-side", type=int, default=0,
                        help="Resize output feature map long side; 0 uses DINO input size")
    parser.add_argument("--upsampler", choices=("bilinear", "jafar"), default="bilinear")
    parser.add_argument("--jafar-repo", default="third_party/JAFAR")
    parser.add_argument("--jafar-ckpt", default=None,
                        help="JAFAR checkpoint. Auto-found from --checkpoint-dir when omitted")
    parser.add_argument("--save-float32", action="store_true",
                        help="Save feature maps as float32 instead of float16")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Debug: process only first N images")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def lazy_imports():
    global Image, ImageOps, np, torch, F, PCA
    from PIL import Image, ImageOps
    import numpy as np
    import torch
    import torch.nn.functional as F
    from sklearn.decomposition import PCA


def torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def list_images(image_dir: Path):
    paths = [p for p in image_dir.iterdir() if p.is_file() and p.suffix in IMAGE_EXTS]
    return sorted(paths, key=lambda p: p.name)


def round_up_to_multiple(x: int, multiple: int) -> int:
    return int(math.ceil(x / multiple) * multiple)


def scaled_size(width: int, height: int, max_long_side: int):
    if max_long_side <= 0 or max(width, height) <= max_long_side:
        return width, height
    scale = max_long_side / float(max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def resize_to_patch_multiple(width: int, height: int, patch_size: int, max_long_side: int):
    width, height = scaled_size(width, height, max_long_side)
    return round_up_to_multiple(width, patch_size), round_up_to_multiple(height, patch_size)


def pil_to_normalized_tensor(image, size_wh, device):
    image = image.convert("RGB").resize(size_wh, Image.BICUBIC)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
    return (tensor - mean) / std


def load_dinov3(args, device):
    repo = Path(args.dinov3_repo).resolve()
    kwargs = {"source": "local"}
    if args.dinov3_weights:
        kwargs["weights"] = args.dinov3_weights
    else:
        auto_weights = auto_find_dinov3_weights(args)
        if auto_weights is not None:
            kwargs["weights"] = str(auto_weights)
            print(f"[DINOv3] auto weights: {auto_weights}")
    print(f"[DINOv3] loading {args.dinov3_model} from {repo}")
    model = torch.hub.load(str(repo), args.dinov3_model, **kwargs)
    model.eval().to(device)
    return model


def auto_find_dinov3_weights(args):
    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.exists():
        return None
    candidates = {
        "dinov3_vits16": ("dinov3_vits16", "vits16"),
        "dinov3_vits16plus": ("dinov3_vits16plus", "vits16plus"),
        "dinov3_vitb16": ("dinov3_vitb16", "vitb16"),
        "dinov3_vitl16": ("dinov3_vitl16", "vitl16"),
        "dinov3_vith16plus": ("dinov3_vith16plus", "vith16plus"),
        "dinov3_vit7b16": ("dinov3_vit7b16", "vit7b16"),
    }.get(args.dinov3_model, (args.dinov3_model,))
    for path in sorted(ckpt_dir.glob("*.pth")):
        name = path.name.lower()
        # Avoid mistaking JAFAR release weights for DINOv3 backbone weights.
        if "patch16_dinov3" in name:
            continue
        if any(token in name for token in candidates) and "dinov3" in name:
            return path
    return None


def model_patch_size(model):
    patch_size = getattr(model, "patch_size", None)
    if isinstance(patch_size, tuple):
        patch_size = patch_size[0]
    if patch_size is None and hasattr(model, "patch_embed"):
        patch_size = getattr(model.patch_embed, "patch_size", None)
        if isinstance(patch_size, tuple):
            patch_size = patch_size[0]
    return int(patch_size or 16)


def load_jafar(args, feature_dim, device):
    if args.upsampler != "jafar":
        return None
    if not args.jafar_ckpt:
        args.jafar_ckpt = str(auto_find_jafar_ckpt(args))
    repo = Path(args.jafar_repo).resolve()
    sys.path.insert(0, str(repo))
    from src.upsampler.jafar import JAFAR

    model = JAFAR(v_dim=feature_dim).to(device).eval()
    ckpt = torch_load(args.jafar_ckpt, map_location="cpu")
    state = ckpt.get("jafar", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
    state = {k.removeprefix("module."): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[JAFAR] missing keys: {len(missing)}")
    if unexpected:
        print(f"[JAFAR] unexpected keys: {len(unexpected)}")
    print(f"[JAFAR] loaded {args.jafar_ckpt}")
    return model


def auto_find_jafar_ckpt(args):
    expected = DINO_TO_JAFAR_CKPT.get(args.dinov3_model)
    if expected is None:
        raise ValueError(
            f"No automatic JAFAR checkpoint mapping for --dinov3-model {args.dinov3_model}. "
            "Pass --jafar-ckpt explicitly."
        )
    path = Path(args.checkpoint_dir) / expected
    if not path.exists():
        raise FileNotFoundError(
            f"Expected JAFAR checkpoint not found: {path}. "
            "Pass --jafar-ckpt explicitly or use --upsampler bilinear."
        )
    print(f"[JAFAR] auto checkpoint: {path}")
    return path


def extract_dense_feature(image_path: Path, dinov3, jafar, args, device, patch_size: int):
    with torch.no_grad():
        image = ImageOps.exif_transpose(Image.open(image_path))
        orig_w, orig_h = image.size
        in_w, in_h = resize_to_patch_multiple(orig_w, orig_h, patch_size, args.max_long_side)
        out_w, out_h = scaled_size(in_w, in_h, args.output_long_side)

        image_tensor = pil_to_normalized_tensor(image, (in_w, in_h), device)
        model_dtype = next(dinov3.parameters()).dtype
        feats = dinov3.get_intermediate_layers(
            image_tensor.to(dtype=model_dtype), n=1, reshape=True, norm=True
        )[0]
        feats = feats.float()
        if not torch.isfinite(feats).all():
            raise RuntimeError(
                f"DINOv3 produced non-finite features for {image_path.name}. "
                "Try --dtype bfloat16 or --dtype float32."
            )

        if jafar is not None:
            dense = jafar(image_tensor.float(), feats, output_size=(out_h, out_w))
        else:
            dense = F.interpolate(feats, size=(out_h, out_w), mode="bilinear", align_corners=False)
        if not torch.isfinite(dense).all():
            raise RuntimeError(
                f"Upsampler produced non-finite features for {image_path.name}. "
                "Try --upsampler bilinear, or check the JAFAR checkpoint/model pairing."
            )
        dense = dense.squeeze(0)
        dense = F.normalize(dense, p=2, dim=0)
        if not torch.isfinite(dense).all():
            raise RuntimeError(f"Feature normalization produced non-finite values for {image_path.name}.")
        return dense.cpu()


def load_or_fit_pca(image_paths, dinov3, jafar, args, device, patch_size):
    pca_path = Path(args.pca_path) if args.pca_path else Path(args.scene).resolve().parent / "dinov3_pca.pth"
    if pca_path.exists() and not args.fit_pca:
        print(f"[PCA] loading {pca_path}")
        payload = torch_load(pca_path, map_location="cpu")
        return payload["pca128"], pca_path

    print("[PCA] collecting samples")
    rng = np.random.default_rng(args.seed)
    all_samples = []
    budget = args.max_pca_samples
    for idx, path in enumerate(image_paths, 1):
        dense = extract_dense_feature(path, dinov3, jafar, args, device, patch_size)
        c, h, w = dense.shape
        pixels = dense.permute(1, 2, 0).reshape(-1, c).numpy()
        finite = np.isfinite(pixels).all(axis=1)
        if not finite.all():
            dropped = int((~finite).sum())
            print(f"[PCA] dropped {dropped} non-finite pixels from {path.name}")
            pixels = pixels[finite]
        n = min(args.samples_per_image, pixels.shape[0], max(0, budget))
        if n <= 0:
            break
        sample_idx = rng.choice(pixels.shape[0], size=n, replace=False)
        all_samples.append(pixels[sample_idx])
        budget -= n
        print(f"[PCA] {idx}/{len(image_paths)} sampled {n} from {path.name}")
        if budget <= 0:
            break

    if not all_samples:
        raise RuntimeError("No PCA samples collected")
    samples = np.concatenate(all_samples, axis=0).astype(np.float32, copy=False)
    sample_var = float(samples.var(axis=0).mean())
    if not np.isfinite(sample_var) or sample_var <= 0:
        raise RuntimeError(
            f"PCA samples have invalid variance ({sample_var}). "
            "This usually means upstream feature extraction collapsed."
        )
    print(f"[PCA] fitting PCA {samples.shape[1]} -> {args.out_dim} on {samples.shape[0]} samples")
    pca = PCA(n_components=args.out_dim, svd_solver="randomized", random_state=args.seed)
    pca.fit(samples)
    pca_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"pca": pca, "pca128": pca, "out_dim": args.out_dim}, pca_path)
    print(f"[PCA] saved {pca_path}")
    return pca, pca_path


def project_feature(dense, pca, chunk: int, save_float32: bool):
    c, h, w = dense.shape
    pixels = dense.permute(1, 2, 0).reshape(-1, c).numpy()
    pixels = np.nan_to_num(pixels, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.empty((pixels.shape[0], pca.n_components_), dtype=np.float32)
    for start in range(0, pixels.shape[0], chunk):
        end = min(start + chunk, pixels.shape[0])
        out[start:end] = pca.transform(pixels[start:end]).astype(np.float32, copy=False)
    out = torch.from_numpy(out).reshape(h, w, pca.n_components_).permute(2, 0, 1).contiguous()
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out = F.normalize(out.float(), p=2, dim=0)
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out if save_float32 else out.half()


def save_feature_maps(image_paths, dinov3, jafar, pca, args, device, patch_size):
    scene = Path(args.scene)
    out_dir = scene / args.output_dir
    f3dgs_dir = scene / args.feature3dgs_dir if args.feature3dgs_dir else None
    out_dir.mkdir(parents=True, exist_ok=True)
    if f3dgs_dir:
        f3dgs_dir.mkdir(parents=True, exist_ok=True)

    for idx, path in enumerate(image_paths, 1):
        stem = path.stem
        gp_path = out_dir / f"{stem}_dinov3_128.pth"
        f3dgs_path = f3dgs_dir / f"{stem}_fmap_CxHxW.pt" if f3dgs_dir else None
        if gp_path.exists() and (not f3dgs_path or f3dgs_path.exists()) and not args.overwrite:
            print(f"[skip] {stem}")
            continue

        dense = extract_dense_feature(path, dinov3, jafar, args, device, patch_size)
        fmap = project_feature(dense, pca, args.transform_chunk, args.save_float32)
        torch.save(fmap, gp_path)
        if f3dgs_path:
            torch.save(fmap, f3dgs_path)
        print(f"[save] {idx}/{len(image_paths)} {stem}: {tuple(fmap.shape)}")


def main():
    args = parse_args()
    lazy_imports()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    scene = Path(args.scene)
    image_dir = scene / args.image_dir
    image_paths = list_images(image_dir)
    if args.limit > 0:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is not available")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    dinov3 = load_dinov3(args, device)
    patch_size = model_patch_size(dinov3)
    print(f"[DINOv3] patch_size={patch_size}, images={len(image_paths)}")

    probe = extract_dense_feature(image_paths[0], dinov3, None, args, device, patch_size)
    feature_dim = int(probe.shape[0])
    del probe
    jafar = load_jafar(args, feature_dim, device)

    if args.dtype == "float16":
        dinov3 = dinov3.half()
    elif args.dtype == "bfloat16":
        dinov3 = dinov3.bfloat16()

    pca, pca_path = load_or_fit_pca(image_paths, dinov3, jafar, args, device, patch_size)
    save_feature_maps(image_paths, dinov3, jafar, pca, args, device, patch_size)
    print(f"\nDone. PCA: {pca_path}")


if __name__ == "__main__":
    main()
