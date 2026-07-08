#!/usr/bin/env bash
set -Eeuo pipefail

# Runs the next comparison set on the prepared multiview batch scenes:
#   1. DINOtxt-1024 + JAFAR, no foreground mask.
#   2. SAM3 plant foreground masks.
#   3. DINOv3 PCA-128 + JAFAR with the SAM3 foreground mask.
#   4. DINOtxt-1024 + JAFAR with the SAM3 foreground mask.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/juxing/.conda/envs/gsplant/bin/python}"
WORK_ROOT="${WORK_ROOT:-$ROOT_DIR/dataset/batch_pinhole_scenes}"

FEATURE_LONG_SIDE="${FEATURE_LONG_SIDE:-512}"
DINOTXT_FEATURE_LONG_SIDE="${DINOTXT_FEATURE_LONG_SIDE:-128}"
FEATURE3DGS_RESOLUTION="${FEATURE3DGS_RESOLUTION:-512}"
FEATURE3DGS_ITERS="${FEATURE3DGS_ITERS:-30000}"
STEP2_ITERS="${STEP2_ITERS:-7000}"
STEP2_RESOLUTION="${STEP2_RESOLUTION:-512}"

MASK_DIR="${MASK_DIR:-sam3_masks_plant}"
SAM3_PROMPTS="${SAM3_PROMPTS:-plant}"
SAM3_CONFIDENCE_THRESHOLD="${SAM3_CONFIDENCE_THRESHOLD:-0.5}"
SAM3_RESOLUTION="${SAM3_RESOLUTION:-1008}"
SAM3_MAX_COMPONENTS="${SAM3_MAX_COMPONENTS:-0}"
SAM3_MORPH_CLOSE="${SAM3_MORPH_CLOSE:-3}"
SAM3_DILATE="${SAM3_DILATE:-2}"
SAM3_MIN_COMPONENT_AREA="${SAM3_MIN_COMPONENT_AREA:-500}"
SKIP_MASKS="${SKIP_MASKS:-0}"

run_batch() {
  env \
    PYTHON="$PYTHON" \
    WORK_ROOT="$WORK_ROOT" \
    FEATURE_LONG_SIDE="$FEATURE_LONG_SIDE" \
    FEATURE3DGS_RESOLUTION="$FEATURE3DGS_RESOLUTION" \
    FEATURE3DGS_ITERS="$FEATURE3DGS_ITERS" \
    STEP2_ITERS="$STEP2_ITERS" \
    STEP2_RESOLUTION="$STEP2_RESOLUTION" \
    FORCE_IMAGES=0 \
    FORCE_COLMAP=0 \
    "$@" \
    bash "$ROOT_DIR/scripts/run_multiview_batch_experiments.sh"
}

run_batch \
  LOG_ROOT="$ROOT_DIR/output/batch_logs_dinotxt_jafar_nomask" \
  DINOTXT_OUT_ROOT="$ROOT_DIR/output/batch_dinotxt_1024_jafar_nomask" \
  RUN_DINOTXT=1 \
  RUN_PCA128_COLOR=0 \
  DINOTXT_TAG=dinotxt_jafar_nomask \
  DINOTXT_OUTPUT_DIR=dinotxt_dim1024_jafar_nomask \
  DINOTXT_FEATURE3DGS_DIR=dinotxt_fmap_jafar_nomask \
  DINOTXT_FEATURE_PRETRAIN=feature_pretrain_dinotxt_jafar_nomask \
  DINOTXT_TEXT_SEED="$WORK_ROOT/text_feats/dinotxt_jafar_nomask_text_feats.pth" \
  DINOTXT_UPSAMPLER=jafar \
  DINOTXT_FEATURE_LONG_SIDE="$DINOTXT_FEATURE_LONG_SIDE" \
  DINOTXT_PLANT_PROMPTS='a photo of a plant|a photo of plant|a close-up photo of a plant' \
  DINO_MASK_DIR= \
  FORCE_FEATURES=0 \
  FORCE_FEATURE3DGS=0 \
  FORCE_ASSETS=0 \
  FORCE_STEP2=0

prompt_args=()
IFS='|' read -ra prompt_items <<< "$SAM3_PROMPTS"
for prompt in "${prompt_items[@]}"; do
  [[ -n "$prompt" ]] && prompt_args+=(--prompt "$prompt")
done

if [[ "$SKIP_MASKS" != "1" ]]; then
  "$PYTHON" "$ROOT_DIR/generate_sam3_plant_masks.py" \
    --scenes-root "$WORK_ROOT" \
    --scene-glob '*_pinhole' \
    --image-dir images \
    --mask-dir "$MASK_DIR" \
    --confidence-threshold "$SAM3_CONFIDENCE_THRESHOLD" \
    --sam3-resolution "$SAM3_RESOLUTION" \
    --max-components "$SAM3_MAX_COMPONENTS" \
    --morph-close "$SAM3_MORPH_CLOSE" \
    --dilate "$SAM3_DILATE" \
    --min-component-area "$SAM3_MIN_COMPONENT_AREA" \
    --empty-policy full \
    "${prompt_args[@]}"
fi

run_batch \
  LOG_ROOT="$ROOT_DIR/output/batch_logs_pca128_color_mask" \
  PCA_OUT_ROOT="$ROOT_DIR/output/batch_pca128_color_mask" \
  RUN_DINOTXT=0 \
  RUN_PCA128_COLOR=1 \
  PCA_TAG=pca128_color_mask \
  PCA_OUTPUT_DIR=dinov3_dim128_mask \
  PCA_FEATURE3DGS_DIR=dinov3_fmap_mask \
  PCA_FEATURE_PRETRAIN=feature_pretrain_pca128_color_mask \
  DINO_MASK_DIR="$MASK_DIR" \
  DINO_MASK_MODE=mean \
  DINO_ZERO_BACKGROUND_FEATURES=1 \
  DINO_PCA_ALL_PIXELS=0 \
  FORCE_FEATURES=0 \
  FORCE_FEATURE3DGS=0 \
  FORCE_ASSETS=0 \
  FORCE_STEP2=0

run_batch \
  LOG_ROOT="$ROOT_DIR/output/batch_logs_dinotxt_jafar_mask" \
  DINOTXT_OUT_ROOT="$ROOT_DIR/output/batch_dinotxt_1024_jafar_mask" \
  RUN_DINOTXT=1 \
  RUN_PCA128_COLOR=0 \
  DINOTXT_TAG=dinotxt_jafar_mask \
  DINOTXT_OUTPUT_DIR=dinotxt_dim1024_jafar_mask \
  DINOTXT_FEATURE3DGS_DIR=dinotxt_fmap_jafar_mask \
  DINOTXT_FEATURE_PRETRAIN=feature_pretrain_dinotxt_jafar_mask \
  DINOTXT_TEXT_SEED="$WORK_ROOT/text_feats/dinotxt_jafar_mask_text_feats.pth" \
  DINOTXT_UPSAMPLER=jafar \
  DINOTXT_FEATURE_LONG_SIDE="$DINOTXT_FEATURE_LONG_SIDE" \
  DINOTXT_PLANT_PROMPTS='a photo of a plant|a photo of plant|a close-up photo of a plant' \
  DINO_MASK_DIR="$MASK_DIR" \
  DINO_MASK_MODE=mean \
  DINO_ZERO_BACKGROUND_FEATURES=1 \
  FORCE_FEATURES=0 \
  FORCE_FEATURE3DGS=0 \
  FORCE_ASSETS=0 \
  FORCE_STEP2=0
