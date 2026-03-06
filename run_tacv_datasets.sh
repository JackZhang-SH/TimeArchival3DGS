#!/bin/bash

# ==========================================
# 1. Global Variables Configuration
# ==========================================
RAW_DATA_DIR="./dataset/soccer_penalty"
TA_DATA_DIR="./dataset/soccer_penalty_3dgs"
OUTPUT_DIR="./output_seq/soccer_penalty"

FRAMES="1-20"
GT_POINT_CLOUD_PATH="/workspace/TimeArchival3DGS/dataset/gt_pc/soccer_penalty/fused_points.ply"

# Terminal Output Colors
CYAN='\033[0;36m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m'

FIRST_FRAME=$(echo $FRAMES | cut -d'-' -f1)

# ==========================================
# 2. Preparation (Convert + Inject GT to FIRST frame ONLY)
# ==========================================
echo -e "${CYAN}>>> [1/2] Converting Data & Injecting GT Point Cloud...${NC}"

if [ -f "convert_to_colmap.py" ]; then
    python convert_to_colmap.py -i "$RAW_DATA_DIR" -o "$TA_DATA_DIR"
    if [ $? -ne 0 ]; then echo -e "${RED}Data conversion failed. Exiting.${NC}"; exit 1; fi
else
    echo -e "${YELLOW}Warning: convert_to_colmap.py not found.${NC}"
fi

# [核心修改]：只将 GT 点云放入第一帧！其他帧坚决不放，保留默认的 1 个点。
echo -e "${YELLOW}>>> Injecting GT point cloud to the FIRST frame ($FIRST_FRAME) ONLY...${NC}"
sparse_dir="$TA_DATA_DIR/frame_$FIRST_FRAME/sparse/0"

if [ -d "$sparse_dir" ] && [ -n "$GT_POINT_CLOUD_PATH" ] && [ -f "$GT_POINT_CLOUD_PATH" ]; then
    rm -f "$sparse_dir/points3D.ply" "$sparse_dir/points3D.bin"
    cp "$GT_POINT_CLOUD_PATH" "$sparse_dir/points3D.ply"
    echo -e "${GREEN}>>> GT Injection Complete for Frame $FIRST_FRAME.${NC}"
else
    echo -e "${YELLOW}>>> Skip GT Injection or files missing.${NC}"
fi

# ==========================================
# 3. Training
# ==========================================
echo -e "\n${CYAN}>>> [2/2] Starting Training Process...${NC}"

# ----------------------------------------------------------------
# Step 3.1: Train the ANCHOR frame (First Frame)
# 只有它使用了 GT 点云，并开启 Densification
# ----------------------------------------------------------------
echo -e "${GREEN}>>> Step 3.1: Training Anchor Frame $FIRST_FRAME (GT Init, Densify ON)${NC}"

python ta_train.py -s "$TA_DATA_DIR" -o "$OUTPUT_DIR" --frames "$FIRST_FRAME" -- \
    --disable_viewer -r 1 --iterations 8000 \
    --densify_until_iter 6000 --opacity_reset_interval 2000 \
    --test_regex "test_.*" \
    --checkpoint_iterations 8000 \
    --white_background

if [ $? -ne 0 ]; then echo -e "${RED}Anchor training failed!${NC}"; exit 1; fi

# ----------------------------------------------------------------
# Step 3.2: Train the CHAIN using native --warm_chain
# --resume-if-exists 自动跳过第一帧，--warm_chain 自动串联后续帧
# ----------------------------------------------------------------
echo -e "\n${YELLOW}>>> Step 3.2: Training Sequence (Warm Chain, Densify OFF)${NC}"

python ta_train.py \
  -s "$TA_DATA_DIR" \
  -o "$OUTPUT_DIR" \
  --frames "$FRAMES" \
  --resume-if-exists \
  --warm_chain \
  -- \
  --disable_viewer -r 1 \
  --iterations 8000 \
  --densify_until_iter 0 --densification_interval 0 \
  --test_regex "test_.*" \
  --checkpoint_iterations 8000 \
  --white_background

if [ $? -ne 0 ]; then echo -e "${RED}Warm chain training failed!${NC}"; exit 1; fi

# ==========================================
# 4. Evaluation
# ==========================================
echo -e "\n${CYAN}>>> Evaluating metrics...${NC}"

python ta_test.py -s "$TA_DATA_DIR" -m "$OUTPUT_DIR" --frames "$FRAMES" \
    --prefix model_frame_ --prefer_model_test_list --read_test_from_model_cfg \
    --save_vis --vis_root "./debug_vis" \
    --white_background

echo -e "\n${GREEN}>>> SUCCESS!${NC}"