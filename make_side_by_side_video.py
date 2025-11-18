#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将两个文件夹中的图片序列分别导出为视频，
再将两路画面左右拼接成一个新视频（画面一半一半）。

用法示例：
    python make_side_by_side_video.py \
        --dir_a path/to/A \
        --dir_b path/to/B \
        --fps 30 \
        --out_a A.mp4 \
        --out_b B.mp4 \
        --out_combined combined.mp4
"""

import os
import glob
import argparse
import cv2


def get_sorted_image_list(folder):
    """获取文件夹下所有图片文件，按文件名排序。"""
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(folder, ext)))
    files.sort()
    return files


def make_video_from_images(image_paths, fps, output_path, frame_size=None):
    """
    将图片序列导出为视频。
    - image_paths: 图片路径列表（已排序）
    - fps: 帧率
    - output_path: 输出视频路径
    - frame_size: (width, height)，如果为 None，则使用第一张图片的尺寸
    """
    if not image_paths:
        raise ValueError(f"No images found for video: {output_path}")

    # 读取第一帧，确定大小
    first_img = cv2.imread(image_paths[0])
    if first_img is None:
        raise RuntimeError(f"Failed to read first image: {image_paths[0]}")

    if frame_size is None:
        h, w = first_img.shape[:2]
        frame_size = (w, h)  # (width, height)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: failed to read {img_path}, skip.")
            continue
        img_resized = cv2.resize(img, frame_size)
        writer.write(img_resized)

    writer.release()
    print(f"Saved video: {output_path}  ({len(image_paths)} frames, size={frame_size})")
    return frame_size  # 返回实际使用的 size


def make_side_by_side_video(image_paths_a, image_paths_b, fps, output_path, frame_size=None):
    """
    从两组图片制作左右拼接的视频。
    - 左边来自 A，右边来自 B。
    - 两边先 resize 成同样大小，然后横向拼接。
    - 最终输出 size ≈ (frame_size[0]*2, frame_size[1])
    """
    if not image_paths_a:
        raise ValueError("No images in dir A for combined video.")
    if not image_paths_b:
        raise ValueError("No images in dir B for combined video.")

    # 使用两组图片数量的较小值，保证 A/B 都有对应帧
    num_frames = min(len(image_paths_a), len(image_paths_b))
    image_paths_a = image_paths_a[:num_frames]
    image_paths_b = image_paths_b[:num_frames]

    # 确定单路画面的大小
    first_a = cv2.imread(image_paths_a[0])
    first_b = cv2.imread(image_paths_b[0])
    if first_a is None:
        raise RuntimeError(f"Failed to read first A image: {image_paths_a[0]}")
    if first_b is None:
        raise RuntimeError(f"Failed to read first B image: {image_paths_b[0]}")

    if frame_size is None:
        # 默认使用 A 的尺寸作为单路画面大小
        ha, wa = first_a.shape[:2]
        frame_size = (wa, ha)

    single_w, single_h = frame_size
    combined_size = (single_w * 2, single_h)  # (width, height)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, combined_size)

    for pa, pb in zip(image_paths_a, image_paths_b):
        img_a = cv2.imread(pa)
        img_b = cv2.imread(pb)
        if img_a is None or img_b is None:
            print(f"Warning: failed to read {pa} or {pb}, skip this frame.")
            continue

        img_a = cv2.resize(img_a, frame_size)
        img_b = cv2.resize(img_b, frame_size)

        # 水平拼接： [A | B]
        combined = cv2.hconcat([img_a, img_b])
        writer.write(combined)

    writer.release()
    print(f"Saved combined video: {output_path}  ({num_frames} frames, size={combined_size})")


def main():
    parser = argparse.ArgumentParser(description="Make two videos from folders and combine them side-by-side.")
    parser.add_argument("--dir_a", required=True, help="文件夹 A，存放第一组图片")
    parser.add_argument("--dir_b", required=True, help="文件夹 B，存放第二组图片")
    parser.add_argument("--fps", type=float, required=True, help="帧率 (fps)")
    parser.add_argument("--out_a", default="A.mp4", help="A 文件夹导出视频路径")
    parser.add_argument("--out_b", default="B.mp4", help="B 文件夹导出视频路径")
    parser.add_argument("--out_combined", default="combined.mp4", help="左右拼接视频输出路径")
    args = parser.parse_args()

    # 获取图片列表
    imgs_a = get_sorted_image_list(args.dir_a)
    imgs_b = get_sorted_image_list(args.dir_b)

    if not imgs_a:
        raise ValueError(f"No images found in dir_a: {args.dir_a}")
    if not imgs_b:
        raise ValueError(f"No images found in dir_b: {args.dir_b}")

    print(f"Found {len(imgs_a)} images in A: {args.dir_a}")
    print(f"Found {len(imgs_b)} images in B: {args.dir_b}")

    # 1. 生成 A 视频（确定一个统一的 frame_size）
    frame_size_a = make_video_from_images(imgs_a, args.fps, args.out_a, frame_size=None)

    # 2. 生成 B 视频，强制使用与 A 相同的尺寸，这样之后左右拼接更稳定
    make_video_from_images(imgs_b, args.fps, args.out_b, frame_size=frame_size_a)

    # 3. 生成左右拼接视频（画面一半一半）
    make_side_by_side_video(imgs_a, imgs_b, args.fps, args.out_combined, frame_size=frame_size_a)


if __name__ == "__main__":
    main()
