<#
.SYNOPSIS
    TA-3DGS Full Pipeline Automation Script with COLMAP Triangulation
#>

<#
.SYNOPSIS
    TA-3DGS Full Pipeline Automation Script with COLMAP Triangulation
#>

# ==========================================
# 1. Global Variables Configuration
# ==========================================
$RAW_DATA_DIR = ".\dataset\Dancing_Walking_Standing\human_multi"
$TA_DATA_DIR  = ".\dataset\human_multi"
$OUTPUT_DIR   = ".\output_seq\human_multi"
$RENDER_DIR   = ".\renders\human_multi"


$FRAMES       = "1-3"  


$REF_FRAME    = "frame_1"
$REF_IMAGE    = "train_xxxx.png"  



# ==========================================
# 2. Pipeline Execution
# ==========================================

Write-Host ">>> [1/5] Converting Instant NGP dataset to COLMAP format..." -ForegroundColor Cyan
python convert_to_colmap.py -i $RAW_DATA_DIR -o $TA_DATA_DIR
if ($LASTEXITCODE -ne 0) { Write-Host "Data conversion failed. Exiting." -ForegroundColor Red; exit }

Write-Host "`n>>> [2/5] Running COLMAP Point Triangulation to generate real point clouds..." -ForegroundColor Cyan

$frame_dirs = Get-ChildItem -Path $TA_DATA_DIR -Directory -Filter "frame_*"
foreach ($dir in $frame_dirs) {
    $frame_path = $dir.FullName
    Write-Host "  -> Running COLMAP for $($dir.Name)..." -ForegroundColor Yellow

    $db_path     = "$frame_path\database.db"
    $img_path    = "$frame_path\images"
    $sparse_dir  = "$frame_path\sparse"
    $sparse_orig = "$sparse_dir\0"
    $sparse_fake = "$sparse_dir\0_fake"
    $sparse_tri  = "$frame_path\sparse_triangulated"


    if (Test-Path $db_path) { Remove-Item $db_path -Force }
    if (Test-Path $sparse_tri) { Remove-Item $sparse_tri -Recurse -Force }
    if (Test-Path $sparse_fake) { Remove-Item $sparse_fake -Recurse -Force }


    colmap feature_extractor --database_path $db_path --image_path $img_path

    colmap exhaustive_matcher --database_path $db_path
    New-Item -ItemType Directory -Force -Path $sparse_tri | Out-Null
    colmap point_triangulator --database_path $db_path --image_path $img_path --input_path $sparse_orig --output_path $sparse_tri

    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [Error] COLMAP Triangulation failed for $($dir.Name). Skipping point cloud swap." -ForegroundColor Red
    } else {
        Rename-Item -Path $sparse_orig -NewName "0_fake"
        Move-Item -Path $sparse_tri -Destination "$sparse_dir\0"
        Write-Host "  [Success] Generated real point cloud and replaced sparse/0 for $($dir.Name)!" -ForegroundColor Green
    }
}

Write-Host "`n>>> [3/5] Starting TA-3DGS batch training (auto-splitting test set)..." -ForegroundColor Cyan
python ta_train.py -s $TA_DATA_DIR -o $OUTPUT_DIR --frames $FRAMES -- --disable_viewer -r 1 --iterations 8000 --densify_until_iter 6000 --opacity_reset_interval 2000 --test_regex "test_.*"
if ($LASTEXITCODE -ne 0) { Write-Host "Training failed. Exiting." -ForegroundColor Red; exit }

Write-Host "`n>>> [4/5] Rendering fixed view ($REF_IMAGE) sequence..." -ForegroundColor Cyan
python ta_render.py -m $OUTPUT_DIR -o $RENDER_DIR --colmap_path "$TA_DATA_DIR\$REF_FRAME" --image_name $REF_IMAGE --frames $FRAMES --prefix model_frame_ --save_format jpeg --jpeg_quality 90 --warmup
if ($LASTEXITCODE -ne 0) { Write-Host "Rendering failed. Exiting." -ForegroundColor Red; exit }

Write-Host "`n>>> [5/5] Evaluating metrics on test set (SSIM/PSNR/LPIPS)..." -ForegroundColor Cyan
python ta_test.py -s $TA_DATA_DIR -m $OUTPUT_DIR --frames $FRAMES --prefix model_frame_ --prefer_model_test_list --read_test_from_model_cfg
if ($LASTEXITCODE -ne 0) { Write-Host "Testing failed. Exiting." -ForegroundColor Red; exit }

Write-Host "`n>>> =========================================" -ForegroundColor Green
Write-Host ">>> SUCCESS: TA-3DGS full pipeline completed!" -ForegroundColor Green
Write-Host ">>> =========================================" -ForegroundColor Green