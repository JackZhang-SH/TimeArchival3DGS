#!/bin/bash


RAW_DATA_DIR="./dataset/Dancing_Walking_Standing/human_multi"
TA_DATA_DIR="./dataset/human_multi"
OUTPUT_DIR="./output_seq/human_multi"


FRAMES="1-3"


CYAN='\033[0;36m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color


echo -e "${CYAN}>>> [1/4] Converting Instant NGP dataset to COLMAP format...${NC}"
python convert_to_colmap.py -i "$RAW_DATA_DIR" -o "$TA_DATA_DIR"
if [ $? -ne 0 ]; then echo -e "${RED}Data conversion failed. Exiting.${NC}"; exit 1; fi

echo -e "\n${CYAN}>>> [2/4] Running COLMAP Point Triangulation to generate real point clouds...${NC}"

for frame_path in "$TA_DATA_DIR"/frame_*; do
    if [ -d "$frame_path" ]; then
        frame_name=$(basename "$frame_path")
        echo -e "${YELLOW}  -> Running COLMAP for $frame_name...${NC}"

        db_path="$frame_path/database.db"
        img_path="$frame_path/images"
        sparse_dir="$frame_path/sparse"
        sparse_orig="$sparse_dir/0"
        sparse_fake="$sparse_dir/0_fake"
        sparse_tri="$frame_path/sparse_triangulated"

        rm -f "$db_path"
        rm -rf "$sparse_tri"
        rm -rf "$sparse_fake"


        colmap feature_extractor --database_path "$db_path" --image_path "$img_path"

        colmap exhaustive_matcher --database_path "$db_path"

        mkdir -p "$sparse_tri"
        colmap point_triangulator --database_path "$db_path" --image_path "$img_path" --input_path "$sparse_orig" --output_path "$sparse_tri"

        if [ $? -ne 0 ]; then
            echo -e "${RED}  [Error] COLMAP Triangulation failed for $frame_name. Skipping point cloud swap.${NC}"
        else
     
            mv "$sparse_orig" "$sparse_fake"
            mv "$sparse_tri" "$sparse_orig"
            echo -e "${GREEN}  [Success] Generated real point cloud and replaced sparse/0 for $frame_name!${NC}"
        fi
    fi
done

echo -e "\n${CYAN}>>> [3/4] Starting TA-3DGS batch training (auto-splitting test set)...${NC}"
python ta_train.py -s "$TA_DATA_DIR" -o "$OUTPUT_DIR" --frames "$FRAMES" -- \
    --disable_viewer -r 1 --iterations 8000 --densify_until_iter 6000 \
    --opacity_reset_interval 2000 --test_regex "test_.*"
if [ $? -ne 0 ]; then echo -e "${RED}Training failed. Exiting.${NC}"; exit 1; fi


echo -e "\n${CYAN}>>> [4/4] Evaluating metrics on test set (SSIM/PSNR/LPIPS)...${NC}"
python ta_test.py -s "$TA_DATA_DIR" -m "$OUTPUT_DIR" --frames "$FRAMES" \
    --prefix model_frame_ --prefer_model_test_list --read_test_from_model_cfg
if [ $? -ne 0 ]; then echo -e "${RED}Testing failed. Exiting.${NC}"; exit 1; fi

echo -e "\n${GREEN}>>> =========================================${NC}"
echo -e "${GREEN}>>> SUCCESS: TA-3DGS full pipeline completed!${NC}"
echo -e "${GREEN}>>> =========================================${NC}"