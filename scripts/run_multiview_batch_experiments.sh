#!/usr/bin/env bash
set -Eeuo pipefail

# Batch pipeline for raw multiview folders under dataset/.
#
# It processes folders named *_multi or *multiview in reverse chronological
# lexical order. For each folder it:
#   1. Creates a PINHOLE COLMAP scene with resized images.
#   2. Runs the native DINOtxt-1024 pipeline.
#   3. Runs the legacy DINOv3 PCA-128 + colour-bootstrap comparison.
#
# Outputs:
#   dataset/batch_pinhole_scenes/<raw_name>_pinhole/
#   output/batch_dinotxt_1024/<raw_name>_pinhole/run/
#   output/batch_pca128_color/<raw_name>_pinhole/run/
#   output/batch_logs/<raw_name>_pinhole/

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/juxing/.conda/envs/gsplant/bin/python}"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/dataset}"
WORK_ROOT="${WORK_ROOT:-$DATA_ROOT/batch_pinhole_scenes}"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/output/batch_logs}"
DINOTXT_OUT_ROOT="${DINOTXT_OUT_ROOT:-$ROOT_DIR/output/batch_dinotxt_1024}"
PCA_OUT_ROOT="${PCA_OUT_ROOT:-$ROOT_DIR/output/batch_pca128_color}"

IMAGE_WIDTH="${IMAGE_WIDTH:-1920}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-1080}"
IMAGE_MODE="${IMAGE_MODE:-crop}"

GPU_INDEX="${GPU_INDEX:-0}"
COLMAP_SINGLE_CAMERA="${COLMAP_SINGLE_CAMERA:-1}"
COLMAP_BA_USE_GPU="${COLMAP_BA_USE_GPU:-1}"

# Keep this conservative for 1024-d feature rendering. Increase for single-scene
# high-quality runs after the batch smoke/validation passes.
FEATURE_LONG_SIDE="${FEATURE_LONG_SIDE:-512}"
FEATURE3DGS_RESOLUTION="${FEATURE3DGS_RESOLUTION:-512}"
FEATURE3DGS_ITERS="${FEATURE3DGS_ITERS:-30000}"
STEP2_ITERS="${STEP2_ITERS:-7000}"
STEP2_RESOLUTION="${STEP2_RESOLUTION:-512}"
STRPR_W_COL="${STRPR_W_COL:-2.0}"
STRPR_W_GEO="${STRPR_W_GEO:-1.0}"
STRPR_W_SEM="${STRPR_W_SEM:-0.5}"
DINO_MASK_DIR="${DINO_MASK_DIR:-}"
DINO_MASK_MODE="${DINO_MASK_MODE:-mean}"
DINO_MASK_THRESHOLD="${DINO_MASK_THRESHOLD:-128}"
DINO_MASK_DILATE="${DINO_MASK_DILATE:-0}"
DINO_MISSING_MASK="${DINO_MISSING_MASK:-error}"
DINO_ZERO_BACKGROUND_FEATURES="${DINO_ZERO_BACKGROUND_FEATURES:-0}"
DINO_PCA_ALL_PIXELS="${DINO_PCA_ALL_PIXELS:-0}"
DINOTXT_TAG="${DINOTXT_TAG:-dinotxt}"
DINOTXT_OUTPUT_DIR="${DINOTXT_OUTPUT_DIR:-dinotxt_dim1024}"
DINOTXT_FEATURE3DGS_DIR="${DINOTXT_FEATURE3DGS_DIR:-dinotxt_fmap}"
DINOTXT_FEATURE_PRETRAIN="${DINOTXT_FEATURE_PRETRAIN:-feature_pretrain_dinotxt}"
DINOTXT_TEXT_SEED="${DINOTXT_TEXT_SEED:-$WORK_ROOT/dinotxt_text_feats.pth}"
DINOTXT_SEMCLS_ROOT="${DINOTXT_SEMCLS_ROOT:-$ROOT_DIR/output/semcls_batch_${DINOTXT_TAG}}"
DINOTXT_UPSAMPLER="${DINOTXT_UPSAMPLER:-bilinear}"
DINOTXT_FEATURE_LONG_SIDE="${DINOTXT_FEATURE_LONG_SIDE:-0}"
DINOTXT_JAFAR_CKPT="${DINOTXT_JAFAR_CKPT:-}"
DINOTXT_LEAF_PROMPTS="${DINOTXT_LEAF_PROMPTS:-}"
DINOTXT_BRANCH_PROMPTS="${DINOTXT_BRANCH_PROMPTS:-}"
DINOTXT_BACKGROUND_PROMPTS="${DINOTXT_BACKGROUND_PROMPTS:-}"
DINOTXT_PLANT_PROMPTS="${DINOTXT_PLANT_PROMPTS:-}"
PCA_TAG="${PCA_TAG:-pca128_color}"
PCA_OUTPUT_DIR="${PCA_OUTPUT_DIR:-dinov3_dim128}"
PCA_FEATURE3DGS_DIR="${PCA_FEATURE3DGS_DIR:-dinov3_fmap}"
PCA_FEATURE_PRETRAIN="${PCA_FEATURE_PRETRAIN:-feature_pretrain_pca128_color}"

RUN_DINOTXT="${RUN_DINOTXT:-1}"
RUN_PCA128_COLOR="${RUN_PCA128_COLOR:-1}"

FORCE_IMAGES="${FORCE_IMAGES:-0}"
FORCE_COLMAP="${FORCE_COLMAP:-0}"
FORCE_FEATURES="${FORCE_FEATURES:-0}"
FORCE_FEATURE3DGS="${FORCE_FEATURE3DGS:-0}"
FORCE_ASSETS="${FORCE_ASSETS:-0}"
FORCE_STEP2="${FORCE_STEP2:-0}"

export CPATH="/usr/local/cuda/include${CPATH:+:$CPATH}"
export CPLUS_INCLUDE_PATH="/usr/local/cuda/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export LIBRARY_PATH="/usr/local/cuda/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"

cd "$ROOT_DIR"
mkdir -p "$WORK_ROOT" "$LOG_ROOT" "$DINOTXT_OUT_ROOT" "$PCA_OUT_ROOT"

log_run() {
  local log_file="$1"
  shift
  mkdir -p "$(dirname "$log_file")"
  echo "[cmd] $*" | tee "$log_file"
  "$@" 2>&1 | tee -a "$log_file"
}

image_count() {
  find "$1" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.tif' -o -iname '*.tiff' \) | wc -l
}

prepare_images() {
  local raw_dir="$1"
  local scene_dir="$2"
  local log_dir="$3"
  local src_count dst_count
  src_count="$(image_count "$raw_dir")"
  dst_count="0"
  [[ -d "$scene_dir/images" ]] && dst_count="$(image_count "$scene_dir/images")"
  if [[ "$FORCE_IMAGES" != "1" && "$src_count" == "$dst_count" && "$dst_count" != "0" ]]; then
    echo "[skip] images already prepared: $scene_dir/images ($dst_count)"
    return
  fi

  log_run "$log_dir/00_prepare_images.log" \
    "$PYTHON" prepare_colmap_images.py \
      --src "$raw_dir" \
      --dst "$scene_dir/images" \
      --width "$IMAGE_WIDTH" \
      --height "$IMAGE_HEIGHT" \
      --mode "$IMAGE_MODE" \
      --overwrite
}

count_registered_images() {
  local model_dir="$1"
  local tmp_dir
  tmp_dir="$(mktemp -d)"
  colmap model_converter \
    --input_path "$model_dir" \
    --output_path "$tmp_dir" \
    --output_type TXT >/dev/null 2>&1
  awk 'BEGIN{n=0} !/^#/ && NF>0 {n++} END{print int(n/2)}' "$tmp_dir/images.txt"
  rm -rf "$tmp_dir"
}

select_largest_model() {
  local models_dir="$1"
  local output_sparse="$2"
  local best_model=""
  local best_count=-1
  local count

  for model_dir in "$models_dir"/*; do
    [[ -d "$model_dir" ]] || continue
    [[ -f "$model_dir/images.bin" || -f "$model_dir/images.txt" ]] || continue
    count="$(count_registered_images "$model_dir")"
    echo "[COLMAP] model $(basename "$model_dir") registered images: $count"
    if (( count > best_count )); then
      best_count="$count"
      best_model="$model_dir"
    fi
  done

  if [[ -z "$best_model" || "$best_count" -le 0 ]]; then
    echo "No valid COLMAP model found in $models_dir" >&2
    exit 1
  fi

  rm -rf "$output_sparse"
  mkdir -p "$output_sparse"
  cp -a "$best_model"/. "$output_sparse"/
  colmap model_converter \
    --input_path "$output_sparse" \
    --output_path "$output_sparse" \
    --output_type TXT >/dev/null 2>&1
  echo "[COLMAP] selected $(basename "$best_model") -> $output_sparse ($best_count registered images)"
}

run_colmap_pinhole() {
  local scene_dir="$1"
  local log_dir="$2"
  local sparse0="$scene_dir/sparse/0"
  local db="$scene_dir/database.db"
  local models_dir="$scene_dir/sparse_models"

  if [[ "$FORCE_COLMAP" != "1" && ( -f "$sparse0/images.bin" || -f "$sparse0/images.txt" ) ]]; then
    echo "[skip] COLMAP already reconstructed: $sparse0"
    return
  fi

  rm -f "$db"
  rm -rf "$models_dir" "$scene_dir/sparse"
  mkdir -p "$models_dir"

  log_run "$log_dir/01_colmap_feature_extractor.log" \
    colmap feature_extractor \
      --database_path "$db" \
      --image_path "$scene_dir/images" \
      --ImageReader.camera_model PINHOLE \
      --ImageReader.single_camera "$COLMAP_SINGLE_CAMERA" \
      --SiftExtraction.use_gpu 1 \
      --SiftExtraction.gpu_index "$GPU_INDEX"

  log_run "$log_dir/02_colmap_exhaustive_matcher.log" \
    colmap exhaustive_matcher \
      --database_path "$db" \
      --SiftMatching.use_gpu 1 \
      --SiftMatching.gpu_index "$GPU_INDEX"

  log_run "$log_dir/03_colmap_mapper.log" \
    colmap mapper \
      --database_path "$db" \
      --image_path "$scene_dir/images" \
      --output_path "$models_dir" \
      --Mapper.ba_use_gpu "$COLMAP_BA_USE_GPU" \
      --Mapper.ba_gpu_index "$GPU_INDEX"

  select_largest_model "$models_dir" "$sparse0" 2>&1 | tee "$log_dir/04_colmap_select_model.log"
}

run_dinotxt_pipeline() {
  local scene_dir="$1"
  local scene_name="$2"
  local log_dir="$3"
  local feature_pretrain="$DINOTXT_FEATURE_PRETRAIN"
  local text_seed="$DINOTXT_TEXT_SEED"
  local text_refine="$WORK_ROOT/text_feats/${scene_name}_${DINOTXT_TAG}_text_feats_semantic_refine.pth"
  local clean_ply="$WORK_ROOT/pretrain_clean/${scene_name}_${DINOTXT_TAG}_clean_pruned.ply"
  local semantic_label="$DINOTXT_SEMCLS_ROOT/$scene_name/${DINOTXT_TAG}_semantic_refine_label.ply"
  local feature_ply="$scene_dir/$feature_pretrain/point_cloud/iteration_${FEATURE3DGS_ITERS}/point_cloud.ply"
  local final_dir="$DINOTXT_OUT_ROOT/$scene_name/run/point_cloud/iteration_${STEP2_ITERS}"
  local mask_args=()
  local upsample_args=(--feature-long-side "$DINOTXT_FEATURE_LONG_SIDE" --upsampler "$DINOTXT_UPSAMPLER")
  local prompt_args=()
  mkdir -p "$WORK_ROOT/text_feats" "$WORK_ROOT/pretrain_clean" "$(dirname "$semantic_label")"
  if [[ -n "$DINO_MASK_DIR" ]]; then
    mask_args+=(--mask-dir "$DINO_MASK_DIR")
    mask_args+=(--mask-mode "$DINO_MASK_MODE")
    mask_args+=(--mask-threshold "$DINO_MASK_THRESHOLD")
    mask_args+=(--mask-dilate "$DINO_MASK_DILATE")
    mask_args+=(--missing-mask "$DINO_MISSING_MASK")
    [[ "$DINO_ZERO_BACKGROUND_FEATURES" == "1" ]] && mask_args+=(--zero-background-features)
  fi
  [[ -n "$DINOTXT_JAFAR_CKPT" ]] && upsample_args+=(--jafar-ckpt "$DINOTXT_JAFAR_CKPT")
  if [[ -n "$DINOTXT_LEAF_PROMPTS" ]]; then
    IFS='|' read -ra prompts <<< "$DINOTXT_LEAF_PROMPTS"
    for prompt in "${prompts[@]}"; do [[ -n "$prompt" ]] && prompt_args+=(--leaf-prompt "$prompt"); done
  fi
  if [[ -n "$DINOTXT_BRANCH_PROMPTS" ]]; then
    IFS='|' read -ra prompts <<< "$DINOTXT_BRANCH_PROMPTS"
    for prompt in "${prompts[@]}"; do [[ -n "$prompt" ]] && prompt_args+=(--branch-prompt "$prompt"); done
  fi
  if [[ -n "$DINOTXT_BACKGROUND_PROMPTS" ]]; then
    IFS='|' read -ra prompts <<< "$DINOTXT_BACKGROUND_PROMPTS"
    for prompt in "${prompts[@]}"; do [[ -n "$prompt" ]] && prompt_args+=(--background-prompt "$prompt"); done
  fi
  if [[ -n "$DINOTXT_PLANT_PROMPTS" ]]; then
    IFS='|' read -ra prompts <<< "$DINOTXT_PLANT_PROMPTS"
    for prompt in "${prompts[@]}"; do [[ -n "$prompt" ]] && prompt_args+=(--plant-prompt "$prompt"); done
  fi

  if [[ "$FORCE_FEATURES" == "1" || ! -d "$scene_dir/$DINOTXT_FEATURE3DGS_DIR" || ! -f "$text_seed" ]]; then
    local overwrite=()
    [[ "$FORCE_FEATURES" == "1" ]] && overwrite=(--overwrite)
    log_run "$log_dir/10_dinotxt_extract.log" \
      "$PYTHON" extract_dinotxt_features.py \
        -s "$scene_dir" \
        --image-dir images \
        --output-dir "$DINOTXT_OUTPUT_DIR" \
        --feature3dgs-dir "$DINOTXT_FEATURE3DGS_DIR" \
        --text-out "$text_seed" \
        --checkpoint-dir checkpoints \
        --max-long-side "$FEATURE_LONG_SIDE" \
        "${upsample_args[@]}" \
        "${mask_args[@]}" \
        "${prompt_args[@]}" \
        "${overwrite[@]}"
  else
    echo "[skip] DINOtxt features already exist: $scene_dir/$DINOTXT_FEATURE3DGS_DIR"
  fi

  if [[ "$FORCE_FEATURE3DGS" == "1" || ! -f "$feature_ply" ]]; then
    (
      cd "$ROOT_DIR/third_party/feature-3dgs"
      log_run "$log_dir/11_dinotxt_feature3dgs.log" \
        "$PYTHON" train.py \
          -s "$scene_dir" \
          -m "$scene_dir/$feature_pretrain" \
          -f dinotxt \
          --semantic_feature_dir "$DINOTXT_FEATURE3DGS_DIR" \
          -r "$FEATURE3DGS_RESOLUTION" \
          --iterations "$FEATURE3DGS_ITERS" \
          --save_iterations 7000 15000 "$FEATURE3DGS_ITERS" \
          --test_iterations 7000 "$FEATURE3DGS_ITERS" \
          --quiet
    )
  else
    echo "[skip] DINOtxt Feature-3DGS exists: $feature_ply"
  fi

  if [[ "$FORCE_ASSETS" == "1" || ! -f "$clean_ply" || ! -f "$text_refine" ]]; then
    log_run "$log_dir/12_dinotxt_prepare_assets.log" \
      "$PYTHON" prepare_step2_assets.py \
        --ply "$feature_ply" \
        --root "$WORK_ROOT" \
        --scene-name "$scene_name" \
        --clean-mode semantic \
        --prototype-mode semantic-refine \
        --seed-text-feats "$text_seed" \
        --text-out "$text_refine" \
        --clean-out "$clean_ply" \
        --semantic-label-out "$semantic_label"
  else
    echo "[skip] DINOtxt Step2 assets exist: $clean_ply"
  fi

  if [[ "$FORCE_STEP2" == "1" || ! -f "$final_dir/branch.ply" ]]; then
    log_run "$log_dir/13_dinotxt_step2.log" \
      "$PYTHON" train.py \
        --source_path "$scene_dir" \
        --root_path "$WORK_ROOT" \
        --model_path "$DINOTXT_OUT_ROOT/$scene_name/run" \
        --clean_ply "$clean_ply" \
        --pretrain_path "$feature_pretrain" \
        --load_iteration "$FEATURE3DGS_ITERS" \
        --mask_path "" \
        --feature_path "" \
        -r "$STEP2_RESOLUTION" \
        --text_feats_path "$text_refine" \
        --label_init semantic \
        --branch_frac -1 \
        --cluster_size 40 \
        --w_col "$STRPR_W_COL" \
        --w_geo "$STRPR_W_GEO" \
        --w_sem "$STRPR_W_SEM" \
        --reg_bind \
        --reg_axis --lambda_axis 1.0 \
        --reg_graph --graph_from 1000 --graph_interval 50 \
        --prune_isolated --prune_from 1500 --prune_interval 500 --prune_until 4000 \
        --densify_branch --branch_split_ratio 1.5 --max_strpr_num 4000 \
        --label_lr 0 \
        --iterations "$STEP2_ITERS" \
        --save_iterations "$STEP2_ITERS" \
        --test_iterations "$STEP2_ITERS" \
        --disable_viewer
  else
    echo "[skip] DINOtxt Step2 exists: $final_dir"
  fi
}

run_pca128_color_pipeline() {
  local scene_dir="$1"
  local scene_name="$2"
  local log_dir="$3"
  local feature_pretrain="$PCA_FEATURE_PRETRAIN"
  local pca_path="$WORK_ROOT/pca/${scene_name}_${PCA_TAG}_dinov3_pca.pth"
  local text_color="$WORK_ROOT/text_feats/${scene_name}_${PCA_TAG}_text_feats.pth"
  local clean_ply="$WORK_ROOT/pretrain_clean/${scene_name}_${PCA_TAG}_clean_pruned.ply"
  local feature_ply="$scene_dir/$feature_pretrain/point_cloud/iteration_${FEATURE3DGS_ITERS}/point_cloud.ply"
  local final_dir="$PCA_OUT_ROOT/$scene_name/run/point_cloud/iteration_${STEP2_ITERS}"
  local mask_args=()
  mkdir -p "$WORK_ROOT/pca" "$WORK_ROOT/text_feats" "$WORK_ROOT/pretrain_clean"
  if [[ -n "$DINO_MASK_DIR" ]]; then
    mask_args+=(--mask-dir "$DINO_MASK_DIR")
    mask_args+=(--mask-mode "$DINO_MASK_MODE")
    mask_args+=(--mask-threshold "$DINO_MASK_THRESHOLD")
    mask_args+=(--mask-dilate "$DINO_MASK_DILATE")
    mask_args+=(--missing-mask "$DINO_MISSING_MASK")
    [[ "$DINO_ZERO_BACKGROUND_FEATURES" == "1" ]] && mask_args+=(--zero-background-features)
    [[ "$DINO_PCA_ALL_PIXELS" == "1" ]] && mask_args+=(--pca-all-pixels)
  fi

  if [[ "$FORCE_FEATURES" == "1" || ! -d "$scene_dir/$PCA_FEATURE3DGS_DIR" || ! -f "$pca_path" ]]; then
    local feature_args=()
    [[ "$FORCE_FEATURES" == "1" ]] && feature_args+=(--overwrite --fit-pca)
    [[ ! -f "$pca_path" ]] && feature_args+=(--fit-pca)
    log_run "$log_dir/20_pca128_extract.log" \
      "$PYTHON" extract_dinov3_features.py \
        -s "$scene_dir" \
        --image-dir images \
        --output-dir "$PCA_OUTPUT_DIR" \
        --feature3dgs-dir "$PCA_FEATURE3DGS_DIR" \
        --checkpoint-dir checkpoints \
        --upsampler jafar \
        --out-dim 128 \
        --pca-path "$pca_path" \
        --max-long-side "$FEATURE_LONG_SIDE" \
        "${mask_args[@]}" \
        "${feature_args[@]}"
  else
    echo "[skip] PCA128 DINOv3 features already exist: $scene_dir/$PCA_FEATURE3DGS_DIR"
  fi

  if [[ "$FORCE_FEATURE3DGS" == "1" || ! -f "$feature_ply" ]]; then
    (
      cd "$ROOT_DIR/third_party/feature-3dgs"
      log_run "$log_dir/21_pca128_feature3dgs.log" \
        "$PYTHON" train.py \
          -s "$scene_dir" \
          -m "$scene_dir/$feature_pretrain" \
          -f dinov3 \
          --semantic_feature_dir "$PCA_FEATURE3DGS_DIR" \
          -r "$FEATURE3DGS_RESOLUTION" \
          --iterations "$FEATURE3DGS_ITERS" \
          --save_iterations 7000 15000 "$FEATURE3DGS_ITERS" \
          --test_iterations 7000 "$FEATURE3DGS_ITERS" \
          --quiet
    )
  else
    echo "[skip] PCA128 Feature-3DGS exists: $feature_ply"
  fi

  if [[ "$FORCE_ASSETS" == "1" || ! -f "$clean_ply" || ! -f "$text_color" ]]; then
    log_run "$log_dir/22_pca128_prepare_assets_color.log" \
      "$PYTHON" prepare_step2_assets.py \
        --ply "$feature_ply" \
        --root "$WORK_ROOT" \
        --scene-name "$scene_name" \
        --text-out "$text_color" \
        --clean-out "$clean_ply"
  else
    echo "[skip] PCA128 colour assets exist: $clean_ply"
  fi

  if [[ "$FORCE_STEP2" == "1" || ! -f "$final_dir/branch.ply" ]]; then
    log_run "$log_dir/23_pca128_step2_color.log" \
      "$PYTHON" train.py \
        --source_path "$scene_dir" \
        --root_path "$WORK_ROOT" \
        --model_path "$PCA_OUT_ROOT/$scene_name/run" \
        --clean_ply "$clean_ply" \
        --pretrain_path "$feature_pretrain" \
        --load_iteration "$FEATURE3DGS_ITERS" \
        --mask_path "" \
        --feature_path "" \
        -r "$STEP2_RESOLUTION" \
        --text_feats_path "$text_color" \
        --label_init joint \
        --branch_frac -1 \
        --cluster_size 40 \
        --w_col "$STRPR_W_COL" \
        --w_geo "$STRPR_W_GEO" \
        --w_sem "$STRPR_W_SEM" \
        --reg_bind \
        --reg_axis --lambda_axis 1.0 \
        --reg_graph --graph_from 1000 --graph_interval 50 \
        --prune_isolated --prune_from 1500 --prune_interval 500 --prune_until 4000 \
        --densify_branch --branch_split_ratio 1.5 --max_strpr_num 4000 \
        --label_lr 0 \
        --iterations "$STEP2_ITERS" \
        --save_iterations "$STEP2_ITERS" \
        --test_iterations "$STEP2_ITERS" \
        --disable_viewer
  else
    echo "[skip] PCA128 colour Step2 exists: $final_dir"
  fi
}

mapfile -t RAW_DATASETS < <(
  find "$DATA_ROOT" -maxdepth 1 -type d \( -name '*multiview' -o -name '*_multi' \) -printf '%f\n' | sort -r
)

if (( ${#RAW_DATASETS[@]} == 0 )); then
  echo "No dataset folders matching *_multi or *multiview found under $DATA_ROOT" >&2
  exit 1
fi

echo "[batch] datasets in descending order:"
printf '  %s\n' "${RAW_DATASETS[@]}"
echo "[batch] work root: $WORK_ROOT"
echo "[batch] DINOtxt output: $DINOTXT_OUT_ROOT"
echo "[batch] PCA128 colour output: $PCA_OUT_ROOT"

for raw_name in "${RAW_DATASETS[@]}"; do
  raw_dir="$DATA_ROOT/$raw_name"
  scene_name="${raw_name}_pinhole"
  scene_dir="$WORK_ROOT/$scene_name"
  log_dir="$LOG_ROOT/$scene_name"
  mkdir -p "$scene_dir" "$log_dir"

  echo
  echo "========== [$scene_name] COLMAP PINHOLE =========="
  prepare_images "$raw_dir" "$scene_dir" "$log_dir"
  run_colmap_pinhole "$scene_dir" "$log_dir"
done

if [[ "$RUN_DINOTXT" == "1" ]]; then
  for raw_name in "${RAW_DATASETS[@]}"; do
    scene_name="${raw_name}_pinhole"
    scene_dir="$WORK_ROOT/$scene_name"
    log_dir="$LOG_ROOT/$scene_name"
    echo
    echo "========== [$scene_name] DINOtxt-1024 =========="
    run_dinotxt_pipeline "$scene_dir" "$scene_name" "$log_dir"
  done
fi

if [[ "$RUN_PCA128_COLOR" == "1" ]]; then
  for raw_name in "${RAW_DATASETS[@]}"; do
    scene_name="${raw_name}_pinhole"
    scene_dir="$WORK_ROOT/$scene_name"
    log_dir="$LOG_ROOT/$scene_name"
    echo
    echo "========== [$scene_name] PCA128 colour bootstrap =========="
    run_pca128_color_pipeline "$scene_dir" "$scene_name" "$log_dir"
  done
fi

echo
echo "[batch] complete"
