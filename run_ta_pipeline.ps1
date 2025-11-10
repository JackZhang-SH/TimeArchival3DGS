param(
  [Parameter(Mandatory = $true)][string]$A_Ply,
  [Parameter(Mandatory = $false)][string]$A_Images_Root = "",
  [Parameter(Mandatory = $false)][string]$A_Images_Single = "",
  [Parameter(Mandatory = $true)][string]$B_Model_Root,
  [Parameter(Mandatory = $true)][string]$B_Dataset_Root,
  [Parameter(Mandatory = $true)][string]$AABB_Json,
  [Parameter(Mandatory = $true)][int]$Frame_Min,
  [Parameter(Mandatory = $true)][int]$Frame_Max,
  [Parameter(Mandatory = $true)][string]$Out_Name,

  # Server params (BindHost overrides ServeHost if provided)
  [string]$ServeHost = "0.0.0.0",
  [string]$BindHost = "",
  [int]$Port = 7860,
  [string]$Camera_Json = "",

  # Stage switches
  [switch]$SkipTrain,
  [switch]$SkipMerge,
  [switch]$SkipPack,
  [switch]$SkipServe
)

$ErrorActionPreference  = "Stop"
$ProgressPreference     = "SilentlyContinue"
$env:PYTHONUNBUFFERED   = "1"   # ensure immediate Python prints

if ($BindHost -and $BindHost.Trim() -ne "") { $ServeHost = $BindHost }

# Runs python with direct console I/O (no redirection), throws on non-zero exit
function Run-Py([string[]]$Argv) {
  Write-Host ">> python $($Argv -join ' ')"
  & python @Argv
  $code = $LASTEXITCODE
  if ($code -ne 0) {
    throw "Command failed (exit=$code): python $($Argv -join ' ')"
  }
}

# Run from script directory
Set-Location -LiteralPath (Split-Path -Parent $MyInvocation.MyCommand.Path)

# Common paths
$OutRoot    = Join-Path (Get-Location) ("output_seq\" + $Out_Name)
$PackedRoot = Join-Path (Get-Location) "output_seq_packed"
$ServeDir   = Join-Path $PackedRoot $Out_Name

# (1) Train
# if (-not $SkipTrain) {
#   Write-Host "========== 1) Train frames $Frame_Min-$Frame_Max (masked AABB) =========="

#   $trainArgv = @(
#     ".\ta_train_masked.py",
#     "-s", $B_Dataset_Root,
#     "-o", $B_Model_Root,
#     "--frames", "$Frame_Min-$Frame_Max",
#     "--mask_mode", "aabb_only",
#     "--aabb", $AABB_Json,
#     "--aabb_close_px", "1", "--close_px", "1", "--dilate_px", "2",
#     "--",
#     "--disable_viewer", "-r", "1", "--optimizer_type", "sparse_adam",
#     "--iterations", "6000", "--position_lr_max_steps", "6000",
#     "--densify_from_iter", "300", "--densify_until_iter", "4800",
#     "--densification_interval", "120", "--densify_grad_threshold", "1.5e-4",
#     "--opacity_reset_interval", "1400", "--lambda_dssim", "0.18",
#     "--percent_dense", "0.012",
#     "--depth_l1_weight_init", "0", "--depth_l1_weight_final", "0",
#     "--test_iterations", "999999",
#     "--save_iterations", "999999"
#   )
#   Run-Py $trainArgv
# } else {
#   Write-Host "[skip] train"
# }

# (2) Merge
if (-not $SkipMerge) {
  Write-Host "========== 2) Merge A + B (multiview mask voting) =========="

  if (-not (Test-Path $OutRoot)) { New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null }

  $mergeArgv = @(
    ".\merge_A_B_batch.py",
    "--a_ply", $A_Ply,
    "--b_model_root", $B_Model_Root,
    "--b_dataset_root", $B_Dataset_Root,
    "--out_root", $OutRoot,
    "--aabb_json", $AABB_Json,
    "--shrink_m", "0.0",
    "--feather_m", "0.0",
    "--cull_outside",
    "--cull_box", "orig",
    "--feature_align", "pad",
    "--mask_ext", ".png",
    "--mask_dilate_px", "0",
    "--min_views", "25",
    "--subsample_cams", "0",
    "--gt_ext", ".png", "--a_ext", ".png",
    "--thr", "25", "--blur_px", "0", "--open_px", "0", "--close_px", "1", "--dilate_px", "5"
  )

  if ($A_Images_Root -and ($A_Images_Root.Trim() -ne "")) {
    $mergeArgv += @("--a_images_root", $A_Images_Root)
  } elseif ($A_Images_Single -and ($A_Images_Single.Trim() -ne "")) {
    $mergeArgv += @("--a_images_single", $A_Images_Single)
  } else {
    Write-Host "[warn] no A images provided; assuming residual masks already exist for each frame."
  }

  Run-Py $mergeArgv
} else {
  Write-Host "[skip] merge"
}

# (3) Pack
if (-not $SkipPack) {
  Write-Host "========== 3) Pack merged models =========="

  if (-not (Test-Path $OutRoot)) {
    throw "Merged root not found: $OutRoot (did you skip merge?)"
  }
  if (-not (Test-Path $PackedRoot)) { New-Item -ItemType Directory -Force -Path $PackedRoot | Out-Null }

  $packArgv = @(
    ".\pack\ta_pack.py",
    "--merged_root", $OutRoot,
    "--name", $Out_Name,
    "--out", $PackedRoot,
    "--autocreate"
  )
  Run-Py $packArgv
} else {
  Write-Host "[skip] pack"
}

# (4) Serve (blocking, full console output; Ctrl+C to stop)
if (-not $SkipServe) {
  Write-Host "========== 4) Serve $ServeDir on $ServeHost`:$Port =========="

  if (-not (Test-Path $ServeDir)) {
    throw "Serve directory not found: $ServeDir (pack output missing?)"
  }

  $serveArgv = @(
    "-u", ".\server\ta_server_slots.py",
    "-p", $ServeDir,
    "--prefix", "model_frame_",
    "--slots", "4",
    "--warmup",
    "--neighbor_prefetch",
    "--host", $ServeHost,
    "--port", "$Port"
  )
  if ($Camera_Json -and ($Camera_Json.Trim() -ne "")) {
    $serveArgv += @("--camera_json", $Camera_Json)
  }

  Run-Py $serveArgv
} else {
  Write-Host "[skip] serve"
}
