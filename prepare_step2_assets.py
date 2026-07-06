#!/usr/bin/env python3
"""Prepare GaussianPlant Step 2 assets from a Feature-3DGS point cloud."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True, help="Feature-3DGS point_cloud.ply")
    parser.add_argument("--root", default="dataset", help="GaussianPlant root_path")
    parser.add_argument("--scene-name", default=None)
    parser.add_argument("--clean-out", default=None)
    parser.add_argument("--text-out", default=None)
    parser.add_argument("--keep-ratio", type=float, default=0.88)
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

    text = torch.stack(
        [
            normalized_mean(feats, green_mask, feats.mean(axis=0)),       # leaf
            normalized_mean(feats, brown_mask, feats.mean(axis=0)),       # branch/stem
            normalized_mean(feats, ~keep, feats.mean(axis=0)),            # background
            normalized_mean(feats, keep, feats.mean(axis=0)),             # plant
        ],
        dim=0,
    )

    text_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"text_feats_dim128": text, "labels": ["leaf", "branch", "background", "plant"]}, text_out)

    clean_out.parent.mkdir(parents=True, exist_ok=True)
    clean_vertex = vertex.data[keep]
    PlyData([PlyElement.describe(clean_vertex, "vertex")], text=ply.text).write(clean_out)

    print(f"[assets] input points: {len(vertex)}")
    print(f"[assets] clean points: {int(keep.sum())} -> {clean_out}")
    print(f"[assets] text feats: {tuple(text.shape)} -> {text_out}")


if __name__ == "__main__":
    main()
