#!/usr/bin/env python3
"""Prepare GaussianPlant Step 2 assets from a Feature-3DGS point cloud."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True, help="Feature-3DGS point_cloud.ply")
    parser.add_argument("--root", default="dataset", help="GaussianPlant root_path")
    parser.add_argument("--scene-name", default=None)
    parser.add_argument("--clean-out", default=None)
    parser.add_argument("--text-out", default=None)
    parser.add_argument("--keep-ratio", type=float, default=0.88)
    parser.add_argument("--prototype-mode", choices=["color", "semantic-refine"], default="color",
                        help="How to build leaf/branch prototypes. color keeps the legacy heuristic; "
                             "semantic-refine classifies with seed prototypes, then rebuilds prototypes "
                             "from semantic high-confidence points.")
    parser.add_argument("--seed-text-feats", default=None,
                        help="Optional seed dinov3_text_feats.pth. Defaults to --text-out if it exists; "
                             "otherwise uses the legacy color prototypes as bootstrap.")
    parser.add_argument("--branch-idx", type=int, default=1)
    parser.add_argument("--leaf-idx", type=int, default=0)
    parser.add_argument("--semantic-tau", type=float, default=0.07)
    parser.add_argument("--semantic-threshold", type=float, default=0.5)
    parser.add_argument("--semantic-margin", type=float, default=0.1,
                        help="High-confidence band around threshold for prototype rebuilding.")
    parser.add_argument("--semantic-label-out", default=None,
                        help="Optional diagnostic PLY with semantic branch probability and colors.")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sh_dc_to_rgb(vertex):
    names = vertex.data.dtype.names
    if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        rgb = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1)
        return np.clip(0.2820948 * rgb + 0.5, 0.0, 1.0)
    if {"red", "green", "blue"}.issubset(names):
        return np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1).astype(np.float32) / 255.0
    return np.full((len(vertex), 3), 0.5, dtype=np.float32)


def semantic_features(vertex):
    names = vertex.data.dtype.names
    sem_names = sorted(
        [n for n in names if n.startswith("semantic_")],
        key=lambda n: int(n.split("_", 1)[1]),
    )
    if not sem_names:
        raise ValueError("No semantic_* properties found in input ply")
    return np.stack([vertex[n] for n in sem_names], axis=1).astype(np.float32)


def normalized_mean(feats, mask, fallback):
    if int(mask.sum()) < 8:
        vec = fallback
    else:
        vec = feats[mask].mean(axis=0)
    vec = torch.from_numpy(vec).float()
    return torch.nn.functional.normalize(vec, dim=0)


def color_seed_prototypes(feats, green_mask, brown_mask, keep):
    fallback = feats.mean(axis=0)
    return torch.stack(
        [
            normalized_mean(feats, green_mask, fallback),       # leaf
            normalized_mean(feats, brown_mask, fallback),       # branch/stem
            normalized_mean(feats, ~keep, fallback),            # background
            normalized_mean(feats, keep, fallback),             # plant
        ],
        dim=0,
    )


def load_seed_text(args, text_out, fallback_text):
    seed_path = Path(args.seed_text_feats) if args.seed_text_feats else text_out
    if seed_path.exists():
        payload = torch.load(seed_path, map_location="cpu", weights_only=False)
        text = payload["text_feats_dim128"].float()
        print(f"[assets] semantic seed prototypes: {seed_path}")
        return text
    print("[assets] semantic seed prototypes: bootstrap from color heuristic")
    return fallback_text


def semantic_refine_prototypes(feats, keep, args, seed_text):
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    clean_feats = torch.from_numpy(feats[keep]).float().to(device)
    visual = F.normalize(clean_feats, dim=-1)
    text = F.normalize(seed_text.float().to(device), dim=-1)
    cos = visual @ text.T
    logits = torch.stack([cos[:, args.leaf_idx], cos[:, args.branch_idx]], dim=1) / args.semantic_tau
    p_branch = F.softmax(logits, dim=1)[:, 1]

    threshold = args.semantic_threshold
    margin = args.semantic_margin
    branch_clean = p_branch >= min(1.0, threshold + margin)
    leaf_clean = p_branch <= max(0.0, threshold - margin)
    if int(branch_clean.sum()) < 8 or int(leaf_clean.sum()) < 8:
        branch_clean = p_branch > threshold
        leaf_clean = ~branch_clean

    all_branch = np.zeros(len(feats), dtype=bool)
    all_leaf = np.zeros(len(feats), dtype=bool)
    keep_idx = np.where(keep)[0]
    all_branch[keep_idx] = branch_clean.detach().cpu().numpy()
    all_leaf[keep_idx] = leaf_clean.detach().cpu().numpy()

    fallback = feats[keep].mean(axis=0)
    refined = torch.stack(
        [
            normalized_mean(feats, all_leaf, fallback),
            normalized_mean(feats, all_branch, fallback),
            F.normalize(seed_text[2].float(), dim=0) if seed_text.shape[0] > 2 else normalized_mean(feats, ~keep, fallback),
            normalized_mean(feats, keep, fallback),
        ],
        dim=0,
    )
    p_all = np.full(len(feats), np.nan, dtype=np.float32)
    p_all[keep_idx] = p_branch.detach().cpu().numpy().astype(np.float32)
    stats = {
        "semantic_branch_fraction": float(branch_clean.float().mean().item()),
        "semantic_leaf_points": int(leaf_clean.sum().item()),
        "semantic_branch_points": int(branch_clean.sum().item()),
        "semantic_threshold": float(threshold),
        "semantic_margin": float(margin),
        "semantic_tau": float(args.semantic_tau),
        "branch_idx": int(args.branch_idx),
        "leaf_idx": int(args.leaf_idx),
    }
    return refined, p_all, stats


def save_semantic_label_ply(path, xyz, p_branch):
    finite = np.isfinite(p_branch)
    is_branch = finite & (p_branch > 0.5)
    rgb = np.zeros((xyz.shape[0], 3), dtype=np.uint8)
    rgb[finite & ~is_branch] = np.array([40, 170, 60], dtype=np.uint8)
    rgb[is_branch] = np.array([220, 40, 40], dtype=np.uint8)
    rgb[~finite] = np.array([90, 90, 90], dtype=np.uint8)
    verts = np.empty(
        xyz.shape[0],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("label", "f4"),
               ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    verts["x"], verts["y"], verts["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    verts["label"] = np.nan_to_num(p_branch, nan=-1.0).astype(np.float32)
    verts["red"], verts["green"], verts["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(verts, "vertex")]).write(path)


def main():
    args = parse_args()
    root = Path(args.root)
    scene_name = args.scene_name or Path(args.ply).parents[2].name
    clean_out = Path(args.clean_out) if args.clean_out else root / "pretrain_clean" / f"{scene_name}_clean_pruned.ply"
    text_out = Path(args.text_out) if args.text_out else root / "dinov3_text_feats.pth"

    ply = PlyData.read(args.ply)
    vertex = ply["vertex"]
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    rgb = sh_dc_to_rgb(vertex)
    feats = semantic_features(vertex)

    brightness = rgb.mean(axis=1)
    saturation = rgb.max(axis=1) - rgb.min(axis=1)
    greenness = rgb[:, 1] - 0.5 * (rgb[:, 0] + rgb[:, 2])
    brownness = rgb[:, 0] - rgb[:, 2] - 0.35 * np.maximum(rgb[:, 1] - rgb[:, 0], 0.0)

    green_mask = greenness >= np.quantile(greenness, 0.58)
    brown_mask = brownness >= np.quantile(brownness, 0.82)
    plant_color = green_mask | brown_mask | (saturation >= np.quantile(saturation, 0.55))

    center = np.median(xyz, axis=0)
    dist = np.linalg.norm(xyz - center, axis=1)
    spatial = dist <= np.quantile(dist, args.keep_ratio)
    z = xyz[:, 2]
    z_crop = (z >= np.quantile(z, 0.01)) & (z <= np.quantile(z, 0.995))
    keep = (plant_color | spatial) & z_crop
    if int(keep.sum()) < max(100, 0.2 * len(keep)):
        keep = spatial & z_crop

    color_text = color_seed_prototypes(feats, green_mask, brown_mask, keep)
    semantic_stats = {}
    if args.prototype_mode == "semantic-refine":
        seed_text = load_seed_text(args, text_out, color_text)
        text, p_branch, semantic_stats = semantic_refine_prototypes(feats, keep, args, seed_text)
        label_out = Path(args.semantic_label_out) if args.semantic_label_out else None
        if label_out is not None:
            save_semantic_label_ply(label_out, xyz, p_branch)
            print(f"[assets] semantic labels: {label_out}")
    else:
        text = color_text

    text_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "text_feats_dim128": text,
        "labels": ["leaf", "branch", "background", "plant"],
        "prototype_mode": args.prototype_mode,
    }
    payload.update(semantic_stats)
    torch.save(payload, text_out)

    clean_out.parent.mkdir(parents=True, exist_ok=True)
    clean_vertex = vertex.data[keep]
    PlyData([PlyElement.describe(clean_vertex, "vertex")], text=ply.text).write(clean_out)

    print(f"[assets] input points: {len(vertex)}")
    print(f"[assets] clean points: {int(keep.sum())} -> {clean_out}")
    print(f"[assets] text feats: {tuple(text.shape)} -> {text_out}")
    if semantic_stats:
        print(f"[assets] semantic branch fraction: {semantic_stats['semantic_branch_fraction']:.3f}")


if __name__ == "__main__":
    main()
