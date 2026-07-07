#!/usr/bin/env python3
"""Extract native 1024-d DINOtxt patch features for GaussianPlant Step 1a.

This script keeps the DINOtxt visual/text alignment intact:

* per-view feature maps are the DINOtxt visual patch tokens, shape [1024, H/16, W/16]
* text prototypes are built from DINOtxt prompts, using the patch-aligned half of
  the 2048-d text embedding, shape [num_classes, 1024]

The feature maps are low-resolution patch grids by default. That keeps 1024-d maps
practical for Feature-3DGS; its renderer already resizes rendered feature maps to the
ground-truth feature map size when computing the feature loss.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".PNG"}
DEFAULT_BPE_URL = "https://dl.fbaipublicfiles.com/dinov3/thirdparty/bpe_simple_vocab_16e6.txt.gz"
DEFAULT_PROMPTS = {
    "leaf": [
        "a photo of a plant leaf",
        "a close-up photo of a plant leaf",
        "a photo of leaves on a plant",
    ],
    "branch": [
        "a photo of a plant stem",
        "a photo of a plant branch",
        "a close-up photo of a woody plant stem",
    ],
    "background": [
        "a photo of the background",
        "a photo of soil, pot, or background",
        "a photo of non-plant background",
    ],
    "plant": [
        "a photo of a plant",
        "a close-up photo of a plant",
        "a photo of plant parts",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Export DINOtxt 1024-d patch feature maps.")
    parser.add_argument("-s", "--scene", required=True, help="Scene folder containing images/")
    parser.add_argument("--image-dir", default="images", help="Image directory relative to scene")
    parser.add_argument("--output-dir", default="dinotxt_dim1024",
                        help="GaussianPlant feature dir relative to scene")
    parser.add_argument("--feature3dgs-dir", default="dinotxt_fmap",
                        help="Feature-3DGS feature dir relative to scene; set empty to disable")
    parser.add_argument("--text-out", default=None,
                        help="Prompt text feature output. Default: <scene_parent>/dinotxt_text_feats.pth")
    parser.add_argument("--dinov3-repo", default="third_party/dinov3")
    parser.add_argument("--dinotxt-weights", default=None,
                        help="DINOtxt vision-head/text-encoder checkpoint")
    parser.add_argument("--backbone-weights", default=None,
                        help="DINOv3 ViT-L backbone checkpoint")
    parser.add_argument("--bpe-vocab", default=None,
                        help="bpe_simple_vocab_16e6.txt.gz path/URL. Defaults to local checkpoint dir or Meta URL")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16",
                        help="Autocast dtype for image feature extraction")
    parser.add_argument("--max-long-side", type=int, default=0,
                        help="Resize image long side before DINOtxt; 0 keeps original size")
    parser.add_argument("--feature-long-side", type=int, default=0,
                        help="Optionally upsample feature grid long side; 0 keeps patch grid")
    parser.add_argument("--save-float32", action="store_true",
                        help="Save feature maps as float32 instead of float16")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--only-text", action="store_true", help="Only write prompt text features")
    parser.add_argument("--limit", type=int, default=0, help="Debug: process only first N images")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--leaf-prompt", action="append", default=None)
    parser.add_argument("--branch-prompt", action="append", default=None)
    parser.add_argument("--background-prompt", action="append", default=None)
    parser.add_argument("--plant-prompt", action="append", default=None)
    return parser.parse_args()


def lazy_imports():
    global Image, ImageOps, np, torch, F
    from PIL import Image, ImageOps
    import numpy as np
    import torch
    import torch.nn.functional as F


def list_images(image_dir: Path):
    paths = [p for p in image_dir.iterdir() if p.is_file() and p.suffix in IMAGE_EXTS]
    return sorted(paths, key=lambda p: p.name)


def scaled_size(width: int, height: int, max_long_side: int):
    if max_long_side <= 0 or max(width, height) <= max_long_side:
        return width, height
    scale = max_long_side / float(max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def find_checkpoint(args, kind: str):
    search_dirs = [Path(args.checkpoint_dir), Path.home() / ".cache/torch/hub/checkpoints"]
    if kind == "dinotxt":
        required = ("dinotxt", "vision_head", "text_encoder")
        excluded = ()
    elif kind == "backbone":
        required = ("dinov3_vitl16", "pretrain_lvd1689m")
        excluded = ("dinotxt", "sat493m")
    else:
        raise ValueError(kind)

    for directory in search_dirs:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.pth")):
            name = path.name.lower()
            if all(token in name for token in required) and not any(token in name for token in excluded):
                return path
    return None


def resolve_bpe_vocab(args):
    if args.bpe_vocab:
        return args.bpe_vocab
    local = Path(args.checkpoint_dir) / "bpe_simple_vocab_16e6.txt.gz"
    if local.exists():
        return str(local)
    return DEFAULT_BPE_URL


def load_dinotxt(args, device):
    repo = Path(args.dinov3_repo).resolve()
    weights = Path(args.dinotxt_weights) if args.dinotxt_weights else find_checkpoint(args, "dinotxt")
    backbone_weights = Path(args.backbone_weights) if args.backbone_weights else find_checkpoint(args, "backbone")
    if weights is None:
        raise FileNotFoundError("DINOtxt weights not found. Pass --dinotxt-weights explicitly.")
    if backbone_weights is None:
        raise FileNotFoundError("DINOv3 ViT-L backbone weights not found. Pass --backbone-weights explicitly.")

    print(f"[DINOtxt] repo: {repo}")
    print(f"[DINOtxt] weights: {weights}")
    print(f"[DINOtxt] backbone: {backbone_weights}")
    print(f"[DINOtxt] bpe: {resolve_bpe_vocab(args)}")
    model, tokenizer = torch.hub.load(
        str(repo),
        "dinov3_vitl16_dinotxt_tet1280d20h24l",
        source="local",
        weights=str(weights),
        backbone_weights=str(backbone_weights),
        bpe_path_or_url=resolve_bpe_vocab(args),
    )
    model.eval().to(device)
    return model, tokenizer


def autocast_context(args, device):
    if device.type != "cuda" or args.dtype == "float32":
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def prompt_sets(args):
    return {
        "leaf": args.leaf_prompt or DEFAULT_PROMPTS["leaf"],
        "branch": args.branch_prompt or DEFAULT_PROMPTS["branch"],
        "background": args.background_prompt or DEFAULT_PROMPTS["background"],
        "plant": args.plant_prompt or DEFAULT_PROMPTS["plant"],
    }


def save_text_features(args, model, tokenizer, device):
    labels = ["leaf", "branch", "background", "plant"]
    prompts = prompt_sets(args)
    class_features = []
    with torch.no_grad():
        for label in labels:
            tokens = tokenizer.tokenize(prompts[label]).to(device, non_blocking=True)
            with autocast_context(args, device):
                feats = model.encode_text(tokens)
            feats = feats[:, feats.shape[1] // 2 :].float()
            feats = F.normalize(feats, p=2, dim=-1)
            feat = F.normalize(feats.mean(dim=0), p=2, dim=0)
            class_features.append(feat.cpu())
            print(f"[text] {label}: {len(prompts[label])} prompts")

    text_feats = torch.stack(class_features, dim=0)
    scene = Path(args.scene).resolve()
    text_out = Path(args.text_out) if args.text_out else scene.parent / "dinotxt_text_feats.pth"
    text_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "text_feats": text_feats,
            "text_feats_dim1024": text_feats,
            "labels": labels,
            "prompt_sets": prompts,
            "feature_type": "dinotxt_patch1024",
        },
        text_out,
    )
    print(f"[text] saved {tuple(text_feats.shape)} -> {text_out}")
    return text_out


def image_to_tensor(image_path: Path, args, device):
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    width, height = scaled_size(*image.size, args.max_long_side)
    image = image.resize((width, height), Image.BICUBIC)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
    return (tensor - mean) / std


def patch_size(model):
    patch = model.visual_model.backbone.patch_size
    return int(patch[0] if isinstance(patch, tuple) else patch)


def extract_patch_feature(image_path: Path, model, args, device):
    image = image_to_tensor(image_path, args, device)
    p = patch_size(model)
    _, _, height, width = image.shape
    new_h = int(math.ceil(height / p) * p)
    new_w = int(math.ceil(width / p) * p)
    if (height, width) != (new_h, new_w):
        image = F.interpolate(image, size=(new_h, new_w), mode="bicubic", align_corners=False)

    with torch.no_grad(), autocast_context(args, device):
        _cls, patch_tokens, _backbone_patch_tokens = model.visual_model.get_class_and_patch_tokens(image)
    feature = patch_tokens.reshape(1, new_h // p, new_w // p, -1).permute(0, 3, 1, 2)[0].float()
    feature = F.normalize(feature, p=2, dim=0)

    if args.feature_long_side > 0:
        _, h, w = feature.shape
        out_w, out_h = scaled_size(w, h, args.feature_long_side)
        feature = F.interpolate(feature.unsqueeze(0), size=(out_h, out_w), mode="bilinear", align_corners=False)[0]
        feature = F.normalize(feature, p=2, dim=0)

    feature = torch.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)
    return feature if args.save_float32 else feature.half()


def save_feature_maps(args, model, device, image_paths):
    scene = Path(args.scene)
    out_dir = scene / args.output_dir
    f3dgs_dir = scene / args.feature3dgs_dir if args.feature3dgs_dir else None
    out_dir.mkdir(parents=True, exist_ok=True)
    if f3dgs_dir:
        f3dgs_dir.mkdir(parents=True, exist_ok=True)

    for idx, image_path in enumerate(image_paths, 1):
        stem = image_path.stem
        gp_path = out_dir / f"{stem}_dinotxt_1024.pth"
        f3dgs_path = f3dgs_dir / f"{stem}_fmap_CxHxW.pt" if f3dgs_dir else None
        if gp_path.exists() and (f3dgs_path is None or f3dgs_path.exists()) and not args.overwrite:
            print(f"[skip] {stem}")
            continue

        feature = extract_patch_feature(image_path, model, args, device)
        torch.save(feature.cpu(), gp_path)
        if f3dgs_path:
            torch.save(feature.cpu(), f3dgs_path)
        print(f"[save] {idx}/{len(image_paths)} {stem}: {tuple(feature.shape)}")


def main():
    args = parse_args()
    lazy_imports()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is not available")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    scene = Path(args.scene)
    image_paths = list_images(scene / args.image_dir)
    if args.limit > 0:
        image_paths = image_paths[: args.limit]
    if not image_paths and not args.only_text:
        raise FileNotFoundError(f"No images found in {scene / args.image_dir}")

    model, tokenizer = load_dinotxt(args, device)
    text_out = save_text_features(args, model, tokenizer, device)
    if not args.only_text:
        print(f"[DINOtxt] patch_size={patch_size(model)}, images={len(image_paths)}")
        save_feature_maps(args, model, device, image_paths)
    print(f"\nDone. Text features: {text_out}")


if __name__ == "__main__":
    main()
