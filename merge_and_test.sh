#!/usr/bin/env bash
# Usage example (override defaults):
# ./merge_and_test.sh \
#   --a-ply output_seq/Static_Point_Cloud/70cams_A_point_cloud.ply \
#   --b-model-root output_seq/soccer_B_48cams \
#   --b-dataset-root dataset/soccer_B_48cams \
#   --out-root output_seq/soccer_merged_48cams \
#   --a-images-single dataset/soccer_A_48cams/frame_1/images \
#   --aabb-json aabb_B_48cams.json

set -e

# ------------------------ Default values ------------------------
A_PLY="output_seq/Static_Point_Cloud/70cams_A_point_cloud.ply"
B_MODEL_ROOT="output_seq/soccer_B_60cams"
B_DATASET_ROOT="dataset/soccer_B_60cams"
OUT_ROOT="output_seq/soccer_merged_60cams"
A_IMAGES_SINGLE="dataset/soccer_A_60cams/frame_1/images"
AABB_JSON="aabb_B.json"
PYTHON_EXE="python"          # You can set to e.g. /path/to/conda/env/python

# Optional: filtered B root (hard-coded in your PS script)
FILTERED_B_ROOT="output_seq/soccer_B_filtered_60cams"

# ------------------------ Parse CLI args ------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --a-ply)
            A_PLY="$2"; shift 2 ;;
        --b-model-root)
            B_MODEL_ROOT="$2"; shift 2 ;;
        --b-dataset-root)
            B_DATASET_ROOT="$2"; shift 2 ;;
        --out-root)
            OUT_ROOT="$2"; shift 2 ;;
        --a-images-single)
            A_IMAGES_SINGLE="$2"; shift 2 ;;
        --aabb-json)
            AABB_JSON="$2"; shift 2 ;;
        --python-exe)
            PYTHON_EXE="$2"; shift 2 ;;
        --filtered-b-root)
            FILTERED_B_ROOT="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "  --a-ply PATH              Static A PLY path"
            echo "  --b-model-root PATH       B model root (output_seq/...)"
            echo "  --b-dataset-root PATH     B dataset root (dataset/...)"
            echo "  --out-root PATH           merged output root"
            echo "  --a-images-single PATH    single-frame A images folder"
            echo "  --aabb-json PATH          AABB json path"
            echo "  --filtered-b-root PATH    filtered B root (optional)"
            echo "  --python-exe PATH         python executable (default: python)"
            exit 0 ;;
        *)
            echo "[WARN] Unknown argument: $1"
            shift ;;
    esac
done

echo "===== Time Archival 3DGS: Merge + Test Pipeline ====="
echo "A_PLY           = $A_PLY"
echo "B_MODEL_ROOT    = $B_MODEL_ROOT"
echo "B_DATASET_ROOT  = $B_DATASET_ROOT"
echo "OUT_ROOT        = $OUT_ROOT"
echo "A_IMAGES_SINGLE = $A_IMAGES_SINGLE"
echo "AABB_JSON       = $AABB_JSON"
echo "FILTERED_B_ROOT = $FILTERED_B_ROOT"
echo "PYTHON_EXE      = $PYTHON_EXE"
echo

# -------------------------------------------------------------------
# Step 1: run merge_A_B_batch.py
# -------------------------------------------------------------------
echo ">>> [Step 1] Merging A + B with merge_A_B_batch.py..."

set +e
"$PYTHON_EXE" merge_A_B_batch.py \
  --a_ply          "$A_PLY" \
  --b_model_root   "$B_MODEL_ROOT" \
  --b_dataset_root "$B_DATASET_ROOT" \
  --out_root       "$OUT_ROOT" \
  --prefix         "model_frame_" \
  --a_images_single "$A_IMAGES_SINGLE" \
  --aabb_json      "$AABB_JSON" \
  --shrink_m       0.0 \
  --feather_m      0.0 \
  --cull_outside \
  --cull_box       "orig" \
  --feature_align  "pad" \
  --mask_ext       ".png" \
  --mask_dilate_px 0 \
  --min_views      15 \
  --subsample_cams 0 \
  --gt_ext         ".png" \
  --a_ext          ".png" \
  --thr            25 \
  --blur_px        0 \
  --open_px        0 \
  --close_px       1 \
  --dilate_px      5 \
  --filtered_b_root "$FILTERED_B_ROOT"
MERGE_EXIT_CODE=$?
set -e

if [[ $MERGE_EXIT_CODE -ne 0 ]]; then
    echo "[ERROR] merge_A_B_batch.py failed (exit code $MERGE_EXIT_CODE)."
    exit $MERGE_EXIT_CODE
fi

echo ">>> merge_A_B_batch.py finished successfully."
echo

# -------------------------------------------------------------------
# Step 2: run ta_test.py on merged scenes
#   -s == B_DATASET_ROOT
#   -m == OUT_ROOT
# -------------------------------------------------------------------
echo ">>> [Step 2] Running ta_test.py on merged 3DGS..."

set +e
"$PYTHON_EXE" ta_test.py \
  -s "$B_DATASET_ROOT" \
  -m "$OUT_ROOT" \
  --frames all \
  --prefix model_frame_ \
  --prefer_model_test_list \
  --read_test_from_model_cfg \
  --sparse_id 0 \
  --iteration -1 \
  --save_vis
TEST_EXIT_CODE=$?
set -e

if [[ $TEST_EXIT_CODE -ne 0 ]]; then
    echo "[ERROR] ta_test.py failed (exit code $TEST_EXIT_CODE)."
    exit $TEST_EXIT_CODE
fi

echo ">>> ta_test.py finished successfully."
echo "===== DONE: Merge + Test pipeline completed. ====="
