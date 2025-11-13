

python .\ta_train.py `
  -s dataset\soccer_B_60cams `
  -o .\output_seq\soccer_B_60cams `
  --frames 1 `
  -- `
  --disable_viewer -r 1 `
  --optimizer_type sparse_adam `
  --iterations 5200 `
  --position_lr_max_steps 5200 `
  --densify_from_iter 300 `
  --densify_until_iter 3600 `
  --densification_interval 180 `
  --densify_grad_threshold 1.6e-4 `
  --percent_dense 0.014 `
  --opacity_reset_interval 1100 `
  --lambda_dssim 0.14 `
  --sh_degree 3 `
  --test_images 0003.png 0030.png

python merge_A_B_batch.py `
  --a_ply Static_Point_Cloud\70cams_A_point_cloud.ply `
  --b_model_root output_seq/soccer_B_60cams `
  --b_dataset_root dataset/soccer_B_60cams `
  --out_root output_seq/soccer_merged_60cams `
  --prefix model_frame_ `
  --a_images_single dataset\soccer_A_60cams\frame_1\images `
  --aabb_json aabb_B.json `
  --shrink_m 0.0 `
  --feather_m 0.0 `
  --cull_outside `
  --cull_box orig `
  --feature_align pad `
  --mask_ext .png `
  --mask_dilate_px 0 `
  --min_views 17 `
  --subsample_cams 0 `
  --gt_ext .png `
  --a_ext .png `
  --thr 25 `
  --blur_px 0 `
  --open_px 0 `
  --close_px 1 `
  --dilate_px 5 `
  --filtered_b_root output_seq/soccer_B_filtered_60cams


python ta_test.py -s dataset/soccer_B_60cams `
  -m output_seq/soccer_merged_60cams  --frames all --prefix model_frame_ `
  --prefer_model_test_list --read_test_from_model_cfg `
  --sparse_id 0 --iteration -1 --save_vis 