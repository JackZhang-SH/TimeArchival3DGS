#.\merge_and_test.ps1 `
# -A_PLY "output_seq\Static_Point_Cloud\70cams_A_point_cloud.ply" `
#  -B_MODEL_ROOT "output_seq/soccer_B_48cams" `
#  -B_DATASET_ROOT "dataset/soccer_B_48cams" `
#  -OUT_ROOT "output_seq/soccer_merged_48cams" `
#  -A_IMAGES_SINGLE "dataset\soccer_A_48cams\frame_1\images" `
#  -AABB_JSON "aabb_B_48cams.json"

param(
    # --------- Core paths (you'll adjust these most often) ---------
    [string]$A_PLY          = "output_seq\Static_Point_Cloud\70cams_A_point_cloud.ply",
    [string]$B_MODEL_ROOT   = "output_seq/soccer_B_60cams",
    [string]$B_DATASET_ROOT = "dataset/soccer_B_60cams",
    [string]$OUT_ROOT       = "output_seq/soccer_merged_60cams",
    [string]$A_IMAGES_SINGLE = "dataset\soccer_A_60cams\frame_1\images",
    [string]$AABB_JSON      = "aabb_B.json",

    # --------- Optional: python executable (if you want to change env) ---------
    [string]$PythonExe      = "python"
)

Write-Host "===== Time Archival 3DGS: Merge + Test Pipeline ====="
Write-Host "A_PLY          = $A_PLY"
Write-Host "B_MODEL_ROOT   = $B_MODEL_ROOT"
Write-Host "B_DATASET_ROOT = $B_DATASET_ROOT"
Write-Host "OUT_ROOT       = $OUT_ROOT"
Write-Host "A_IMAGES_SINGLE= $A_IMAGES_SINGLE"
Write-Host "AABB_JSON      = $AABB_JSON"
Write-Host ""

# -------------------------------------------------------------------
# Step 1: run merge_A_B_batch.py
# -------------------------------------------------------------------
Write-Host ">>> [Step 1] Merging A + B with merge_A_B_batch.py..." -ForegroundColor Cyan

& $PythonExe .\merge_A_B_batch.py `
  --a_ply         "$A_PLY" `
  --b_model_root  "$B_MODEL_ROOT" `
  --b_dataset_root "$B_DATASET_ROOT" `
  --out_root      "$OUT_ROOT" `
  --prefix        "model_frame_" `
  --a_images_single "$A_IMAGES_SINGLE" `
  --aabb_json     "$AABB_JSON" `
  --shrink_m      0.0 `
  --feather_m     0.0 `
  --cull_outside `
  --cull_box      "orig" `
  --feature_align "pad" `
  --mask_ext      ".png" `
  --mask_dilate_px 0 `
  --min_views     15 `
  --subsample_cams 0 `
  --gt_ext        ".png" `
  --a_ext         ".png" `
  --thr           25 `
  --blur_px       0 `
  --open_px       0 `
  --close_px      1 `
  --dilate_px     5 `
  --filtered_b_root "output_seq/soccer_B_filtered_60cams"

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] merge_A_B_batch.py failed (exit code $LASTEXITCODE)." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ">>> merge_A_B_batch.py finished successfully." -ForegroundColor Green
Write-Host ""

# -------------------------------------------------------------------
# Step 2: run ta_test.py on merged scenes
#   -s == B_DATASET_ROOT
#   -m == OUT_ROOT
# -------------------------------------------------------------------
Write-Host ">>> [Step 2] Running ta_test.py on merged 3DGS..." -ForegroundColor Cyan

& $PythonExe .\ta_test.py `
  -s "$B_DATASET_ROOT" `
  -m "$OUT_ROOT" `
  --frames all `
  --prefix model_frame_ `
  --prefer_model_test_list `
  --read_test_from_model_cfg `
  --sparse_id 0 `
  --iteration -1 `
  --save_vis

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] ta_test.py failed (exit code $LASTEXITCODE)." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ">>> ta_test.py finished successfully." -ForegroundColor Green
Write-Host "===== DONE: Merge + Test pipeline completed. ====="
