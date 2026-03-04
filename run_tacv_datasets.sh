#!/bin/bash

# ==========================================
# 1. Global Variables Configuration
# ==========================================
RAW_DATA_DIR="./dataset/human_multi"
TA_DATA_DIR="./dataset/human_multi_3dgs"
OUTPUT_DIR="./output_seq/human_multi"

FRAMES="1-5"
GT_POINT_CLOUD_PATH="/workspace/TimeArchival3DGS/dataset/gt_pc/human/fused_points.ply"
USE_COLMAP="false"

# Terminal Output Colors
CYAN='\033[0;36m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m'

FIRST_FRAME=$(echo $FRAMES | cut -d'-' -f1)
LAST_FRAME=$(echo $FRAMES | cut -d'-' -f2)

# ==========================================
# 2. Preparation
# ==========================================
echo -e "${CYAN}>>> [1/2] Converting Data & Injecting GT Point Cloud...${NC}"

if [ -f "convert_to_colmap.py" ]; then
    python convert_to_colmap.py -i "$RAW_DATA_DIR" -o "$TA_DATA_DIR"
    if [ $? -ne 0 ]; then echo -e "${RED}Data conversion failed. Exiting.${NC}"; exit 1; fi
fi

echo -e "${YELLOW}>>> Injecting GT point cloud to ALL frames...${NC}"
for (( i=FIRST_FRAME; i<=LAST_FRAME; i++ ))
do
    sparse_dir="$TA_DATA_DIR/frame_$i/sparse/0"
    if [ -d "$sparse_dir" ]; then
        if [ -n "$GT_POINT_CLOUD_PATH" ] && [ -f "$GT_POINT_CLOUD_PATH" ]; then
            rm -f "$sparse_dir/points3D.ply" "$sparse_dir/points3D.bin"
            cp "$GT_POINT_CLOUD_PATH" "$sparse_dir/points3D.ply"
        fi
    fi
done
echo -e "${GREEN}>>> Injection Complete.${NC}"

# ==========================================
# 3. Training Loop
# ==========================================
echo -e "\n${CYAN}>>> [2/2] Starting Training Loop...${NC}"

for (( i=FIRST_FRAME; i<=LAST_FRAME; i++ ))
do
    echo -e "\n----------------------------------------------------------------"
    
    if [ "$i" -eq "$FIRST_FRAME" ]; then
        echo -e "${GREEN}>>> Training Frame $i (ANCHOR)${NC}"
        python ta_train.py -s "$TA_DATA_DIR" -o "$OUTPUT_DIR" --frames "$i" -- \
            --disable_viewer -r 1 --iterations 8000 \
            --densify_until_iter 6000 --opacity_reset_interval 2000 \
            --test_regex "test_.*" \
            --checkpoint_iterations 8000 --white_background
    else
        PREV=$((i - 1))
        PREV_CKPT="$OUTPUT_DIR/model_frame_${PREV}/chkpnt8000.pth"
        echo -e "${YELLOW}>>> Training Frame $i (CHAIN from $PREV)${NC}"
        
        if [ ! -f "$PREV_CKPT" ]; then echo -e "${RED}Chain broken!${NC}"; exit 1; fi
        
        # [修改] 移除了 --white_background
        python ta_train.py -s "$TA_DATA_DIR" -o "$OUTPUT_DIR" --frames "$i" -- \
            --disable_viewer -r 1 --iterations 8000 \
            --densify_until_iter 0 --densification_interval 0 \
            --test_regex "test_.*" \
            --checkpoint_iterations 8000 \
            --start_checkpoint "$PREV_CKPT" \
            --reset_start_iter --white_background
    fi
    if [ $? -ne 0 ]; then exit 1; fi
done

echo -e "\n${CYAN}>>> Evaluating metrics...${NC}"
# [修改] 移除了 --white_background
python ta_test.py -s "$TA_DATA_DIR" -m "$OUTPUT_DIR" --frames "$FRAMES" \
    --prefix model_frame_ --prefer_model_test_list --read_test_from_model_cfg --save_vis --vis_root "./debug_vis" --white_background
echo -e "\n${GREEN}>>> SUCCESS!${NC}"