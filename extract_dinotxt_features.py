#!/usr/bin/env python3
"""Extract native 1024-d DINOtxt patch features for GaussianPlant Step 1a.

This script keeps the DINOtxt visual/text alignment intact:

* per-view feature maps are the DINOtxt visual patch tokens, shape [1024, H/16, W/16]
* text prototypes are built from DINOtxt prompts, using the patch-aligned half of
  the 2048-d text embedding, shape [num_classes, 1024]

The feature maps are low-resolution patch grids by default. That keeps 1024-d maps
practical for Feature-3DGS; its renderer already resizes rendered feature maps to the
ground-truth feature map size when computing the feature loss. If denser supervision
is needed, the patch grid can optionally be upsampled with JAFAR.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".PNG"}
MASK_EXTS = (".JPG", ".jpg", ".png", ".PNG", ".jpeg", ".JPEG", ".tif", ".tiff")
IMAGENET_MEAN_RGB = (124, 116, 104)
DINOTXT_FEATURE_DIM = 1024
DINOTXT_JAFAR_CKPT = "vit_large_patch16_dinov3.lvd1689m.pth"
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
    parser.add_argument("--mask-dir", default="",
                        help="Optional foreground mask dir relative to scene, e.g. masks")
    parser.add_argument("--mask-mode", choices=("mean", "black", "white", "blur", "fgmean"),
                        default="mean",
                        help="How to replace masked-out background before DINOtxt")
    parser.add_argument("--mask-threshold", type=int, default=128)
    parser.add_argument("--mask-dilate", type=int, default=0,
                        help="Dilate foreground mask by this many pixels before applying it")
    parser.add_argument("--mask-blur-radius", type=float, default=25.0,
                        help="Gaussian blur radius for --mask-mode blur")
    parser.add_argument("--missing-mask", choices=("error", "ignore"), default="error",
                        help="What to do when --mask-dir is set but an image mask is missing")
    parser.add_argument("--zero-background-features", action="store_true",
                        help="Set saved feature vectors outside the foreground mask to zero")
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
                        help="Force output feature grid long side; 0 keeps patch grid")
    parser.add_argument("--upsampler", choices=("bilinear", "jafar"), default="bilinear",
                        help="Upsampler used when --feature-long-side > 0")
    parser.add_argument("--jafar-repo", default="third_party/JAFAR")
    parser.add_argument("--jafar-ckpt", default=None,
                        help="JAFAR checkpoint. Auto-found from --checkpoint-dir when omitted")
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
    global Image, ImageFilter, ImageOps, np, torch, F
    from PIL import Image, ImageFilter, ImageOps
    import numpy as np
    import torch
    import torch.nn.functional as F


def list_images(image_dir: Path):
    paths = [p for p in image_dir.iterdir() if p.is_file() and p.suffix in IMAGE_EXTS]
    return sorted(paths, key=lambda p: p.name)


def torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def resolve_mask_path(image_path: Path, args):
    if not args.mask_dir:
        return None
    root = Path(args.mask_dir)
    if not root.is_absolute():
        root = Path(args.scene) / root
    candidates = [root / image_path.name]
    candidates.extend(root / f"{image_path.stem}{ext}" for ext in MASK_EXTS)
    for path in candidates:
        if path.exists():
            return path
    if args.missing_mask == "ignore":
        return None
    raise FileNotFoundError(f"Mask not found for {image_path.name} in {root}")


def load_mask_array(image_path: Path, args, size_wh):
    mask_path = resolve_mask_path(image_path, args)
    if mask_path is None:
        return None
    mask = Image.open(mask_path).convert("L").resize(size_wh, Image.NEAREST)
    if args.mask_dilate > 0:
        mask = mask.filter(ImageFilter.MaxFilter(args.mask_dilate * 2 + 1))
    return np.asarray(mask) > args.mask_threshold


def apply_foreground_mask(image_path: Path, image, args):
    image = image.convert("RGB")
    mask = load_mask_array(image_path, args, image.size)
    if mask is None:
        return image

    arr = np.asarray(image, dtype=np.uint8)
    if args.mask_mode == "black":
        background = np.zeros_like(arr)
    elif args.mask_mode == "white":
        background = np.full_like(arr, 255)
    elif args.mask_mode == "blur":
        background = np.asarray(
            image.filter(ImageFilter.GaussianBlur(args.mask_blur_radius)),
            dtype=np.uint8,
        )
    elif args.mask_mode == "fgmean" and mask.any():
        fill = np.round(arr[mask].mean(axis=0)).astype(np.uint8)
        background = np.broadcast_to(fill, arr.shape).copy()
    else:
        fill = np.array(IMAGENET_MEAN_RGB, dtype=np.uint8)
        background = np.broadcast_to(fill, arr.shape).copy()
    out = np.where(mask[..., None], arr, background)
    return Image.fromarray(out.astype(np.uint8), mode="RGB")


def feature_mask_for_image(image_path: Path, args, feature_size_wh):
    if not args.mask_dir:
        return None
    return load_mask_array(image_path, args, feature_size_wh)


def scaled_size(width: int, height: int, max_long_side: int):
    if max_long_side <= 0 or max(width, height) <= max_long_side:
        return width, height
    scale = max_long_side / float(max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def target_long_side_size(width: int, height: int, target_long_side: int):
    if target_long_side <= 0:
        return width, height
    scale = target_long_side / float(max(width, height))
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


def auto_find_jafar_ckpt(args):
    path = Path(args.checkpoint_dir) / DINOTXT_JAFAR_CKPT
    if not path.exists():
        raise FileNotFoundError(
            f"Expected DINOtxt/JAFAR checkpoint not found: {path}. "
            "Pass --jafar-ckpt explicitly or use --upsampler bilinear."
        )
    print(f"[JAFAR] auto checkpoint: {path}")
    return path


def load_jafar(args, feature_dim, device):
    if args.upsampler != "jafar":
        return None
    if args.feature_long_side <= 0:
        raise ValueError("--upsampler jafar requires --feature-long-side > 0")
    if not args.jafar_ckpt:
        args.jafar_ckpt = str(auto_find_jafar_ckpt(args))

    repo = Path(args.jafar_repo).resolve()
    sys.path.insert(0, str(repo))
    from src.upsampler import jafar as jafar_module

    original_create_coordinate = jafar_module.create_coordinate

    def create_coordinate_on_device(h, w, start=0, end=1, device=device, dtype=torch.float32):
        return original_create_coordinate(h, w, start=start, end=end, device=device, dtype=dtype)

    jafar_module.create_coordinate = create_coordinate_on_device
    JAFAR = jafar_module.JAFAR

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
    image = apply_foreground_mask(image_path, image, args)
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


def extract_patch_feature(image_path: Path, model, jafar, args, device):
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
        out_w, out_h = target_long_side_size(w, h, args.feature_long_side)
        if jafar is not None:
            with torch.no_grad():
                feature = jafar(image.float(), feature.unsqueeze(0), output_size=(out_h, out_w))[0].float()
        else:
            feature = F.interpolate(feature.unsqueeze(0), size=(out_h, out_w), mode="bilinear", align_corners=False)[0]
        feature = F.normalize(feature, p=2, dim=0)

    feature = torch.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)
    if args.zero_background_features:
        _, h, w = feature.shape
        foreground_mask = feature_mask_for_image(image_path, args, (w, h))
        if foreground_mask is not None:
            mask_t = torch.from_numpy(foreground_mask.astype(bool)).to(feature.device)
            feature = feature * mask_t.unsqueeze(0).to(feature.dtype)
    return feature if args.save_float32 else feature.half()


def save_feature_maps(args, model, jafar, device, image_paths):
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

        feature = extract_patch_feature(image_path, model, jafar, args, device)
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
        jafar = load_jafar(args, DINOTXT_FEATURE_DIM, device)
        print(
            f"[DINOtxt] patch_size={patch_size(model)}, images={len(image_paths)}, "
            f"upsampler={args.upsampler}, feature_long_side={args.feature_long_side}"
        )
        save_feature_maps(args, model, jafar, device, image_paths)
    print(f"\nDone. Text features: {text_out}")


if __name__ == "__main__":
    main()
