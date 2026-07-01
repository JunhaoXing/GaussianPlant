#!/usr/bin/env python3
"""Compile/warm up gsplat CUDA kernels with a tiny synthetic scene.

Run this once inside the GaussianPlant environment before the first real train:

    python compile_gsplat_dummy.py

It exercises the same two rasterization shapes used by gaussian_renderer:
RGB+expected-depth with SH colors, and raw semantic-feature rasterization.
"""

from __future__ import annotations

import argparse
import math
import sys
import time


def make_camera(width: int, height: int, device: torch.device):
    fx = width / (2.0 * math.tan(math.radians(50.0) * 0.5))
    fy = height / (2.0 * math.tan(math.radians(50.0) * 0.5))
    viewmats = torch.eye(4, device=device, dtype=torch.float32)[None]
    Ks = torch.tensor(
        [[fx, 0.0, width * 0.5], [0.0, fy, height * 0.5], [0.0, 0.0, 1.0]],
        device=device,
        dtype=torch.float32,
    )[None]
    return viewmats, Ks


def make_gaussians(num_points: int, sh_degree: int, semantic_dim: int, device: torch.device):
    means = torch.tensor(
        [
            [-0.12, -0.06, 1.10],
            [0.10, -0.04, 1.25],
            [-0.02, 0.12, 1.40],
            [0.16, 0.12, 1.65],
        ],
        device=device,
        dtype=torch.float32,
    )
    if num_points != means.shape[0]:
        gen = torch.Generator(device=device).manual_seed(1234)
        xy = 0.22 * (torch.rand((num_points, 2), device=device, generator=gen) - 0.5)
        z = 1.0 + 0.8 * torch.rand((num_points, 1), device=device, generator=gen)
        means = torch.cat([xy, z], dim=1)
    means.requires_grad_(True)

    quats = torch.zeros((num_points, 4), device=device, dtype=torch.float32)
    quats[:, 0] = 1.0
    quats.requires_grad_(True)

    scales = torch.full((num_points, 3), 0.045, device=device, dtype=torch.float32)
    scales[:, 2] = 0.08
    scales.requires_grad_(True)

    opacities = torch.full((num_points,), 0.65, device=device, dtype=torch.float32)
    opacities.requires_grad_(True)

    sh_channels = (sh_degree + 1) ** 2
    sh_colors = torch.zeros((num_points, sh_channels, 3), device=device, dtype=torch.float32)
    sh_colors[:, 0, :] = torch.tensor([0.45, 0.75, 0.35], device=device)
    if sh_channels > 1:
        sh_colors[:, 1:, :] = 0.02 * torch.randn(
            (num_points, sh_channels - 1, 3), device=device
        )
    sh_colors.requires_grad_(True)

    semantic_colors = torch.randn((num_points, semantic_dim), device=device, dtype=torch.float32)
    semantic_colors.requires_grad_(True)

    return means, quats, scales, opacities, sh_colors, semantic_colors


def run_pass(
    rasterization,
    *,
    width: int,
    height: int,
    sh_degree: int,
    semantic_dim: int,
    num_points: int,
    device: torch.device,
    backward: bool,
):
    viewmats, Ks = make_camera(width, height, device)
    means, quats, scales, opacities, sh_colors, semantic_colors = make_gaussians(
        num_points, sh_degree, semantic_dim, device
    )

    rgb_bg = torch.zeros((1, 3), device=device, dtype=torch.float32)
    t0 = time.perf_counter()
    rgb_depth, _, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=sh_colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        sh_degree=sh_degree,
        render_mode="RGB+ED",
        backgrounds=rgb_bg,
        packed=False,
        absgrad=False,
    )
    if backward:
        rgb_depth.square().mean().backward()
    torch.cuda.synchronize(device)
    print(f"[ok] RGB+ED SH pass: {time.perf_counter() - t0:.2f}s")

    for tensor in (means, quats, scales, opacities, sh_colors):
        tensor.grad = None

    feature_bg = torch.zeros((1, semantic_dim), device=device, dtype=torch.float32)
    t0 = time.perf_counter()
    features, _, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=semantic_colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        sh_degree=None,
        render_mode="RGB",
        backgrounds=feature_bg,
        packed=False,
        absgrad=False,
    )
    if backward:
        features.square().mean().backward()
    torch.cuda.synchronize(device)
    print(f"[ok] semantic feature pass: {time.perf_counter() - t0:.2f}s")


def parse_args():
    parser = argparse.ArgumentParser(description="Warm up gsplat JIT compilation.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--num-points", type=int, default=4)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--semantic-dim", type=int, default=128)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--no-backward", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        global torch
        import torch
        from gsplat import rasterization
        import gsplat
    except Exception as exc:  # noqa: BLE001
        print(f"[error] Could not import torch/gsplat: {exc}", file=sys.stderr)
        print("Install them first in the GaussianPlant environment.", file=sys.stderr)
        return 1

    if not torch.cuda.is_available():
        print("[error] CUDA is not available; gsplat kernels cannot be compiled.", file=sys.stderr)
        return 1

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    print(f"torch: {torch.__version__}")
    print(f"gsplat: {getattr(gsplat, '__version__', 'unknown')}")
    print(f"device: {torch.cuda.get_device_name(device)} ({device})")

    for i in range(args.repeat):
        print(f"\nWarmup pass {i + 1}/{args.repeat}")
        run_pass(
            rasterization,
            width=args.width,
            height=args.height,
            sh_degree=args.sh_degree,
            semantic_dim=args.semantic_dim,
            num_points=args.num_points,
            device=device,
            backward=not args.no_backward,
        )

    print("\nDone. gsplat kernels used by GaussianPlant should now be compiled/cached.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
