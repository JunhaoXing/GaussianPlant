#!/usr/bin/env bash
set -Eeuo pipefail

# Full DINOtxt-1024 pipeline for dataset/20251204_scene_pinhole.
# Run from anywhere:
#   bash scripts/run_dinotxt_1024_20251204_scene_pinhole.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/juxing/.conda/envs/gsplant/bin/python}"

SCENE_NAME="${SCENE_NAME:-20251204_scene_pinhole}"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/dataset}"
SCENE_PATH="$DATA_ROOT/$SCENE_NAME"

DINO_MAX_LONG_SIDE="${DINO_MAX_LONG_SIDE:-1024}"
FEATURE3DGS_ITERS="${FEATURE3DGS_ITERS:-30000}"
STEP2_ITERS="${STEP2_ITERS:-7000}"
STEP2_RESOLUTION="${STEP2_RESOLUTION:-512}"

FEATURE_PRETRAIN_DIR="feature_pretrain_dinotxt"
MODEL_PATH="$ROOT_DIR/output/$SCENE_NAME/run_dinotxt_1024"
LOG_DIR="$ROOT_DIR/output/$SCENE_NAME/logs_dinotxt_1024"

TEXT_FEATS="$DATA_ROOT/dinotxt_text_feats.pth"
TEXT_FEATS_REFINE="$DATA_ROOT/dinotxt_text_feats_semantic_refine.pth"
CLEAN_PLY="$DATA_ROOT/pretrain_clean/${SCENE_NAME}_clean_pruned.ply"
SEMANTIC_LABEL_PLY="$ROOT_DIR/output/semcls/$SCENE_NAME/dinotxt_semantic_refine_label.ply"
FEATURE_PLY="$SCENE_PATH/$FEATURE_PRETRAIN_DIR/point_cloud/iteration_${FEATURE3DGS_ITERS}/point_cloud.ply"

mkdir -p "$LOG_DIR" "$ROOT_DIR/output/semcls/$SCENE_NAME" "$DATA_ROOT/pretrain_clean"

export CPATH="/usr/local/cuda/include${CPATH:+:$CPATH}"
export CPLUS_INCLUDE_PATH="/usr/local/cuda/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export LIBRARY_PATH="/usr/local/cuda/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"

cd "$ROOT_DIR"

echo "[0/4] Root: $ROOT_DIR"
echo "[0/4] Scene: $SCENE_PATH"
echo "[0/4] Logs: $LOG_DIR"

echo "[1/4] Extracting native DINOtxt-1024 feature maps"
"$PYTHON" extract_dinotxt_features.py \
  -s "$SCENE_PATH" \
  --image-dir images \
  --output-dir dinotxt_dim1024 \
  --feature3dgs-dir dinotxt_fmap \
  --checkpoint-dir checkpoints \
  --max-long-side "$DINO_MAX_LONG_SIDE" \
  --overwrite \
  2>&1 | tee "$LOG_DIR/step1a_extract_dinotxt.log"

echo "[2/4] Training Feature-3DGS with DINOtxt features"
(
  cd "$ROOT_DIR/third_party/feature-3dgs"
  "$PYTHON" train.py \
    -s "$SCENE_PATH" \
    -m "$SCENE_PATH/$FEATURE_PRETRAIN_DIR" \
    -f dinotxt \
    -r 0 \
    --iterations "$FEATURE3DGS_ITERS" \
    --save_iterations 7000 15000 "$FEATURE3DGS_ITERS" \
    --test_iterations 7000 "$FEATURE3DGS_ITERS" \
    --quiet
) 2>&1 | tee "$LOG_DIR/step1b_feature3dgs_dinotxt.log"

if [[ ! -f "$FEATURE_PLY" ]]; then
  echo "Missing Feature-3DGS output: $FEATURE_PLY" >&2
  exit 1
fi

echo "[3/4] Preparing semantic-clean Step2 assets"
"$PYTHON" prepare_step2_assets.py \
  --ply "$FEATURE_PLY" \
  --root "$DATA_ROOT" \
  --scene-name "$SCENE_NAME" \
  --clean-mode semantic \
  --prototype-mode semantic-refine \
  --seed-text-feats "$TEXT_FEATS" \
  --text-out "$TEXT_FEATS_REFINE" \
  --clean-out "$CLEAN_PLY" \
  --semantic-label-out "$SEMANTIC_LABEL_PLY" \
  2>&1 | tee "$LOG_DIR/step1c_prepare_step2_assets.log"

echo "[4/4] Training GaussianPlant Step2"
"$PYTHON" train.py \
  --source_path "$SCENE_PATH" \
  --root_path "$DATA_ROOT" \
  --model_path "$MODEL_PATH" \
  --clean_ply "pretrain_clean/${SCENE_NAME}_clean_pruned.ply" \
  --pretrain_path "$FEATURE_PRETRAIN_DIR" \
  --load_iteration "$FEATURE3DGS_ITERS" \
  --mask_path "" \
  --feature_path "" \
  -r "$STEP2_RESOLUTION" \
  --text_feats_path "$(basename "$TEXT_FEATS_REFINE")" \
  --label_init semantic \
  --branch_frac -1 \
  --cluster_size 40 \
  --reg_bind \
  --reg_axis --lambda_axis 1.0 \
  --reg_graph --graph_from 1000 --graph_interval 50 \
  --prune_isolated --prune_from 1500 --prune_interval 500 --prune_until 4000 \
  --densify_branch --branch_split_ratio 1.5 --max_strpr_num 4000 \
  --label_lr 0 \
  --iterations "$STEP2_ITERS" \
  --save_iterations "$STEP2_ITERS" \
  --test_iterations "$STEP2_ITERS" \
  --disable_viewer \
  2>&1 | tee "$LOG_DIR/step2_gaussianplant.log"

FINAL_DIR="$MODEL_PATH/point_cloud/iteration_${STEP2_ITERS}"

echo "Done."
echo "Feature-3DGS output: $FEATURE_PLY"
echo "Clean PLY: $CLEAN_PLY"
echo "Refined text feats: $TEXT_FEATS_REFINE"
echo "GaussianPlant output: $FINAL_DIR"
