
Param(
  [Parameter(Mandatory=$true)] [string] $A_Ply,
  [Parameter(Mandatory=$true)] [string] $B_Model_Root,
  [Parameter(Mandatory=$true)] [string] $B_Dataset_Root,
  [Parameter(Mandatory=$true)] [string] $AABB_Json,
  [Parameter(Mandatory=$true)] [int]    $Frame_Min,
  [Parameter(Mandatory=$true)] [int]    $Frame_Max,
  [Parameter(Mandatory=$true)] [string] $Out_Name,
  [int]    $Slots = 4,
  [int]    $Port = 7860,
  [string] $BindHost = "0.0.0.0",
  [string] $Camera_Json = "..\camera.json",
  [switch] $NoViewer  # if specified, DO NOT pass --disable_viewer to train
)

# Fail-fast
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# Go to script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

function Get-FullPath([string]$p) {
  if ([string]::IsNullOrWhiteSpace($p)) { return $null }
  # Use Path.GetFullPath so it doesn't require the path to already exist
  return [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $p))
}

$A_Ply          = Get-FullPath $A_Ply
$B_Model_Root   = Get-FullPath $B_Model_Root
$B_Dataset_Root = Get-FullPath $B_Dataset_Root
$AABB_Json      = Get-FullPath $AABB_Json
$Out_Seq        = Join-Path $ScriptDir "output_seq"
$Out_Root       = Join-Path $Out_Seq $Out_Name
$Packed_Base    = Join-Path $ScriptDir "output_seq_packed"
$Packed_Root    = Join-Path $Packed_Base $Out_Name

# Helpers
function Step([string]$Title, [scriptblock]$Block) {
  Write-Host "========== $Title ==========" -ForegroundColor Cyan
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  & $Block
  $sw.Stop()
  Write-Host "[OK] $Title  (${($sw.Elapsed.ToString())})" -ForegroundColor Green
}

function Run-Cmd([string]$Cmd) {
  Write-Host ">> $Cmd" -ForegroundColor Yellow
  $psi = New-Object System.Diagnostics.ProcessStartInfo
  $psi.FileName = "powershell.exe"
  $psi.Arguments = "-NoProfile -ExecutionPolicy Bypass -Command `"${Cmd}`""
  $psi.UseShellExecute = $false
  $psi.RedirectStandardOutput = $true
  $psi.RedirectStandardError = $true
  $proc = [System.Diagnostics.Process]::Start($psi)
  $stdout = $proc.StandardOutput.ReadToEnd()
  $stderr = $proc.StandardError.ReadToEnd()
  $proc.WaitForExit()
  if ($stdout) { Write-Host $stdout }
  if ($stderr) { Write-Host $stderr -ForegroundColor DarkYellow }
  if ($proc.ExitCode -ne 0) {
    throw "Command failed (exit=$($proc.ExitCode)): $Cmd"
  }
}

# 1) Train
$framesArg = "$Frame_Min-$Frame_Max"
$trainArgs = @(
  "python .\ta_train_masked.py",
  "-s `"$B_Dataset_Root`"",
  "-o `"$B_Model_Root`"",
  "--frames $framesArg",
  "--mask_mode aabb_only",
  "--aabb `"$AABB_Json`"",
  "--save_mask_png",
  "--aabb_close_px 1 --close_px 1 --dilate_px 2",
  "--masked",
  "--",
  "--iterations 6000 -r 1 --sh_degree 3",
  "--densify_from_iter 800 --densify_until_iter 3200",
  "--densification_interval 400 --densify_grad_threshold 5e-4",
  "--opacity_reset_interval 10000 --lambda_dssim 0.10",
  "--depth_l1_weight_init 0 --depth_l1_weight_final 0",
  "--test_iterations 999999 --save_iterations 999999",
  "--optimizer_type sparse_adam"
)
if (-not $NoViewer) { $trainArgs += "--disable_viewer" }
$cmdTrain = ($trainArgs -join " ")

Step "1) Train frames $framesArg (masked AABB)" { Run-Cmd $cmdTrain }

# 2) Merge
$mergeArgs = @(
  "python .\merge_A_B_batch.py",
  "--a_ply `"$A_Ply`"",
  "--b_root `"$B_Model_Root`"",
  "--out_root `"$Out_Root`"",
  "--feature_align pad",
  "--mask_ext .png --mask_dilate_px 0 --min_views 25 --subsample_cams 0"
)
$cmdMerge = ($mergeArgs -join " ")
Step "2) Merge A + B (mask voting) → $Out_Root" { Run-Cmd $cmdMerge }

# 3) Pack merged
$packArgs = @(
  "python .\ta_pack.py",
  "--merged_root `"$Out_Root`"",
  "--name `"$Out_Name`"",
  "--out `"$Packed_Base`"",
  "--autocreate"
)
$cmdPack = ($packArgs -join " ")
Step "3) Pack merged → $Packed_Root" { Run-Cmd $cmdPack }

# 4) Serve
$serveArgs = @(
  "python .\server\ta_server_slots.py",
  "-p `"$Packed_Root`"",
  "--prefix model_frame_",
  "--slots $Slots --warmup --neighbor_prefetch",
  "--camera_json `"$Camera_Json`"",
  "--host $BindHost --port $Port"
)
$cmdServe = ($serveArgs -join " ")
# escape colon in title to avoid $var: parsing
$hostPort = "$($BindHost)`:$Port"
Step "4) Serve $Packed_Root on $hostPort" { Run-Cmd $cmdServe }
