#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import math

import torch
import torch.nn.functional as F
from gsplat import rasterization

from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh


def _view_and_K(viewpoint_camera):
    device = viewpoint_camera.world_view_transform.device
    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1)
    width = int(viewpoint_camera.image_width)
    height = int(viewpoint_camera.image_height)
    fx = width / (2.0 * math.tan(viewpoint_camera.FoVx * 0.5))
    fy = height / (2.0 * math.tan(viewpoint_camera.FoVy * 0.5))
    K = torch.tensor(
        [[fx, 0.0, width * 0.5], [0.0, fy, height * 0.5], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    return viewmat[None], K[None], width, height


def _radii_to_scalar(radii):
    if radii.dim() == 3:
        radii = radii.amax(dim=-1)
    return radii.squeeze(0)


def _colors_from_sh(viewpoint_camera, pc, pipe, override_color):
    if override_color is not None:
        return override_color, None

    if pipe.convert_SHs_python:
        shs_view = pc.get_features.transpose(1, 2).view(
            -1, 3, (pc.max_sh_degree + 1) ** 2
        )
        dirs = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
        dirs = dirs / dirs.norm(dim=1, keepdim=True)
        return torch.clamp_min(eval_sh(pc.active_sh_degree, shs_view, dirs) + 0.5, 0.0), None

    return pc.get_features, pc.active_sh_degree


def _rasterize(
    viewpoint_camera,
    means,
    quats,
    scales,
    opacities,
    colors,
    bg_color,
    sh_degree,
    render_mode,
    scaling_modifier=1.0,
):
    viewmats, Ks, width, height = _view_and_K(viewpoint_camera)
    out, _alphas, info = rasterization(
        means=means,
        quats=quats,
        scales=scales * scaling_modifier,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        sh_degree=sh_degree,
        render_mode=render_mode,
        backgrounds=bg_color[None] if bg_color is not None else None,
        packed=False,
        absgrad=False,
    )
    return out, info


def _prepare(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color):
    means = pc.get_xyz
    quats = pc.get_rotation
    scales = pc.get_scaling
    opacities = pc.get_opacity.squeeze(-1)
    colors, sh_degree = _colors_from_sh(viewpoint_camera, pc, pipe, override_color)
    return means, quats, scales, opacities, colors, bg_color, sh_degree, scaling_modifier


def calculate_selection_score(features, query_features, score_threshold=None, positive_ids=[0]):
    features = F.normalize(features.float(), p=2, dim=-1)
    query_features = F.normalize(query_features.float(), p=2, dim=-1)
    scores = features @ query_features.T
    if scores.shape[-1] == 1:
        scores = scores[:, 0]
        scores = (scores >= score_threshold).float()
    else:
        scores = torch.nn.functional.softmax(scores, dim=-1)
        if score_threshold is not None:
            scores = scores[:, positive_ids].sum(-1)
            scores = (scores >= score_threshold).float()
        else:
            scores[:, positive_ids[0]] = scores[:, positive_ids].sum(-1)
            scores = torch.isin(
                torch.argmax(scores, dim=-1),
                torch.tensor(positive_ids, device=features.device),
            ).float()
    return scores


def calculate_selection_score_delete(features, query_features, score_threshold=None, positive_ids=[0]):
    features = F.normalize(features.float(), p=2, dim=-1)
    query_features = F.normalize(query_features.float(), p=2, dim=-1)
    scores = features @ query_features.T
    if scores.shape[-1] == 1:
        scores = scores[:, 0]
        mask = (scores >= score_threshold).float()
    else:
        scores = torch.nn.functional.softmax(scores, dim=-1)
        scores[:, positive_ids[0]] = scores[:, positive_ids].sum(-1)
        mask = torch.isin(
            torch.argmax(scores, dim=-1),
            torch.tensor(positive_ids, device=features.device),
        )
        if score_threshold is not None:
            threshold_scores = scores[:, positive_ids].sum(-1)
            mask = torch.bitwise_or(threshold_scores >= score_threshold, mask)
        mask = mask.float()
    return mask


def _render_from_prepared(
    viewpoint_camera,
    means,
    quats,
    scales,
    opacities,
    colors,
    bg_color,
    sh_degree,
    scaling_modifier,
    semantic_feature,
):
    out, info = _rasterize(
        viewpoint_camera,
        means,
        quats,
        scales,
        opacities,
        colors,
        bg_color,
        sh_degree,
        render_mode="RGB+ED",
        scaling_modifier=scaling_modifier,
    )
    rendered_image = out[0, ..., :3].permute(2, 0, 1)
    depth = out[0, ..., 3:4].permute(2, 0, 1)

    means2d = info["means2d"]
    if means2d.requires_grad:
        means2d.retain_grad()
    radii = _radii_to_scalar(info["radii"])

    if semantic_feature is not None and semantic_feature.numel() > 0:
        features = semantic_feature.squeeze(1)
        feature_bg = torch.zeros(features.shape[1], dtype=features.dtype, device=features.device)
        feature_out, _ = _rasterize(
            viewpoint_camera,
            means,
            quats,
            scales,
            opacities,
            features,
            feature_bg,
            sh_degree=None,
            render_mode="RGB",
            scaling_modifier=scaling_modifier,
        )
        feature_map = feature_out[0].permute(2, 0, 1)
    else:
        feature_map = torch.empty(
            0,
            int(viewpoint_camera.image_height),
            int(viewpoint_camera.image_width),
            dtype=rendered_image.dtype,
            device=rendered_image.device,
        )

    return {
        "render": rendered_image,
        "viewspace_points": means2d,
        "visibility_filter": radii > 0,
        "radii": radii,
        "feature_map": feature_map,
        "depth": depth,
    }


def render_edit(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    text_feature: torch.Tensor,
    edit_dict: dict,
    scaling_modifier=1.0,
    override_color=None,
):
    means, quats, scales, opacities, colors, bg_color, sh_degree, scaling_modifier = _prepare(
        viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color
    )
    opacities = opacities.clone()
    colors = colors.clone()

    semantic_feature = pc.get_semantic_feature
    positive_ids = edit_dict["positive_ids"]
    score_threshold = edit_dict["score_threshold"]
    op_dict = edit_dict["operations"]

    if "deletion" in op_dict:
        scores = calculate_selection_score_delete(
            semantic_feature[:, 0, :], text_feature, score_threshold, positive_ids
        )
        opacities.masked_fill_(scores >= 0.5, 0)
    if "extraction" in op_dict:
        scores = calculate_selection_score(
            semantic_feature[:, 0, :], text_feature, score_threshold, positive_ids
        )
        opacities.masked_fill_(scores <= 0.5, 0)
    if "color_func" in op_dict and sh_degree is not None:
        scores = calculate_selection_score(
            semantic_feature[:, 0, :], text_feature, score_threshold, positive_ids
        )
        colors[:, 0, :] = (
            colors[:, 0, :] * (1 - scores[:, None])
            + op_dict["color_func"](colors[:, 0, :]) * scores[:, None]
        )

    return _render_from_prepared(
        viewpoint_camera,
        means,
        quats,
        scales,
        opacities,
        colors,
        bg_color,
        sh_degree,
        scaling_modifier,
        semantic_feature,
    )


def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
):
    prepared = _prepare(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color)
    return _render_from_prepared(viewpoint_camera, *prepared, pc.get_semantic_feature)
