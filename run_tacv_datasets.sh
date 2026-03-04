#!/bin/bash

# ==========================================
# 1. Global Variables Configuration
# ==========================================
RAW_DATA_DIR="./dataset/human_multi"
TA_DATA_DIR="./dataset/human_multi_3dgs"
OUTPUT_DIR="./output_seq/human_multi"

# Format rule: Must be "start-end" (e.g., "3-6") or a single frame (e.g., "3")
FRAMES="1-5"

# --- 🌟 New Feature: GT Point Cloud Injection 🌟 ---
GT_POINT_CLOUD_PATH="/workspace/TimeArchival3DGS/dataset/gt_pc/human/fused_points.ply"

# --- 🌟 Control Switch 🌟 ---
USE_COLMAP="false"

# Terminal Output Colors
CYAN='\033[0;36m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

FIRST_FRAME=$(echo $FRAMES | cut -d'-' -f1)
LAST_FRAME=$(echo $FRAMES | cut -d'-' -f2)

# ==========================================
# 2. Pipeline Execution
# ==========================================

echo -e "${CYAN}>>> [1/4] Converting Instant NGP dataset to COLMAP format (Auto White BG)...${NC}"
if [ -f "convert_to_colmap.py" ]; then
    python convert_to_colmap.py -i "$RAW_DATA_DIR" -o "$TA_DATA_DIR"
    if [ $? -ne 0 ]; then echo -e "${RED}Data conversion failed. Exiting.${NC}"; exit 1; fi
else
    echo -e "${YELLOW}Warning: convert_to_colmap.py not found, skipping conversion step.${NC}"
fi

echo -e "\n${CYAN}>>> [2/4] Preparing Point Cloud for FIRST Frame ($FIRST_FRAME)...${NC}"

frame_name="frame_${FIRST_FRAME}"
frame_path="$TA_DATA_DIR/$frame_name"
sparse_dir="$frame_path/sparse/0"

if [ ! -d "$frame_path" ]; then
    echo -e "${RED}  [Error] $frame_path not found! Make sure conversion succeeded.${NC}"
    exit 1
fi

if [ -n "$GT_POINT_CLOUD_PATH" ] && [ -f "$GT_POINT_CLOUD_PATH" ]; then
    echo -e "${YELLOW}  -> GT Point Cloud detected at: $GT_POINT_CLOUD_PATH${NC}"
    echo -e "${YELLOW}  -> Injecting GT points into $frame_name (Skipping COLMAP)...${NC}"

    mkdir -p "$sparse_dir"
    
    # [修改] 保留 points3D.txt 供 COLMAP GUI 调试使用，只删除旧的二进制/PLY
    rm -f "$sparse_dir/points3D.ply" "$sparse_dir/points3D.bin"

    filename=$(basename -- "$GT_POINT_CLOUD_PATH")
    extension="${filename##*.}"
    
    if [ "$extension" == "ply" ]; then
        cp "$GT_POINT_CLOUD_PATH" "$sparse_dir/points3D.ply"
    elif [ "$extension" == "bin" ]; then
        cp "$GT_POINT_CLOUD_PATH" "$sparse_dir/points3D.bin"
    else
        cp "$GT_POINT_CLOUD_PATH" "$sparse_dir/points3D.ply"
    fi

    echo -e "${GREEN}  [Success] GT Point Cloud injected successfully!${NC}"

elif [ "$USE_COLMAP" = "true" ]; then
    echo -e "${YELLOW}  -> Running GPU COLMAP for $frame_name...${NC}"
    # (COLMAP logic omitted)
    # ...
else
    echo -e "${YELLOW}  -> Skipping COLMAP entirely, using random points for the first frame.${NC}"
fi

echo -e "\n${CYAN}>>> [3/4] Starting TA-3DGS Training...${NC}"

# ==========================================
# 🌟 Phase 3.1: Train the FIRST frame (with Densification)
# ==========================================
echo -e "${YELLOW}  -> Phase 3.1: Training FIRST frame ($FIRST_FRAME) with standard densification...${NC}"

# [修复核心 BUG]: 手动添加 --checkpoint_iterations 8000
# 这样 Frame 1 才会保存 .pth 文件，供 Frame 2 读取
python ta_train.py -s "$TA_DATA_DIR" -o "$OUTPUT_DIR" --frames "$FIRST_FRAME" -- \
    --disable_viewer -r 1 --iterations 8000 --densify_until_iter 6000 \
    --opacity_reset_interval 2000 --test_regex "test_.*" \
    --white_background \
    --checkpoint_iterations 8000

if [ $? -ne 0 ]; then echo -e "${RED}Training first frame failed. Exiting.${NC}"; exit 1; fi

# ==========================================
# 🌟 Phase 3.2: Train subsequent frames (Warm Chain ON, Densification OFF)
# ==========================================
if [ "$FIRST_FRAME" != "$LAST_FRAME" ] && [ -n "$LAST_FRAME" ]; then
    REST_FRAMES="$((FIRST_FRAME+1))-$LAST_FRAME"
    echo -e "${YELLOW}  -> Phase 3.2: Training subsequent frames ($REST_FRAMES) using Warm Start & NO Densification...${NC}"
    
    # 这里 ta_train.py 的 --warm_chain 会自动添加 checkpoint_iterations，但手动加也没坏处
    python ta_train.py -s "$TA_DATA_DIR" -o "$OUTPUT_DIR" --frames "$REST_FRAMES" \
        --resume-if-exists --warm_chain -- \
        --disable_viewer -r 1 --iterations 8000 \
        --densify_until_iter 0 --densification_interval 0 \
        --test_regex "test_.*" \
        --white_background
        
    if [ $? -ne 0 ]; then echo -e "${RED}Training subsequent frames failed. Exiting.${NC}"; exit 1; fi
fi

echo -e "\n${CYAN}>>> [4/4] Evaluating metrics on test set (SSIM/PSNR/LPIPS)...${NC}"
python ta_test.py -s "$TA_DATA_DIR" -m "$OUTPUT_DIR" --frames "$FRAMES" \
    --prefix model_frame_ --prefer_model_test_list --read_test_from_model_cfg

if [ $? -ne 0 ]; then echo -e "${RED}Testing failed. Exiting.${NC}"; exit 1; fi

echo -e "\n${GREEN}>>> =========================================${NC}"
echo -e "${GREEN}>>> SUCCESS: TA-3DGS pipeline completed!${NC}"
echo -e "${GREEN}>>> =========================================${NC}"