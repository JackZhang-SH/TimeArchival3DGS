# Time Archival 3DGS (TA-3DGS)

TA-3DGS is a **3D Gaussian Splatting (3DGS)** variant of  **[TACV (Time-Archival Camera Virtualization)](https://github.com/JackZhang-SH/Time-Archival-Camera-Virtualization-for-Visual-Performance-and-Sports)**: a dynamic sports scene is represented as a **time-indexed set of multi-view snapshots**, and we train / render / evaluate **per time step (per frame folder)**.
**Paper status:** TACV manuscript under revision at *Computer Vision and Image Understanding (CVIU)*.
Code is released now; a paper link/preprint will be added when publicly available.
**Preprint:** Not publicly available yet.


This repository is intended to pair nicely with **[RS Studio](https://github.com/JackZhang-SH/RealSynth_Studio)** exports (COLMAP + 3DGS Surface Points), i.e., **COLMAP-style frames + `fused_points.ply`** for convenient initialization and training.


![GT (left) vs Render (right)](./assets/demo.gif)

*Left: GT video | Right: Render video*



---

## Per-frame training + evaluation (stable)

### What "stable" means here
The stable path focuses on a robust baseline workflow:
- **Train one 3DGS per frame folder** (`frame_1`, `frame_2`, ...)
- **Render / evaluate per frame**
- **(Optional) Pack models** for faster loading
- **(Optional) Run the TA server GUI** to browse frames

> The **A+B merge** workflow is **NOT** part of the stable quickstart. See the **Experimental** section below.

---

## Installation (tested on RunPod Linux)
**Note:** This setup assumes an NVIDIA driver compatible with CUDA 12.1 (check `nvidia-smi`).  
If your driver/CUDA differs, install the matching PyTorch wheel.

Below is the exact setup I tested on a RunPod Linux server.

```bash
apt-get update -y
apt-get install -y wget bzip2 git build-essential zip unzip
git clone https://github.com/JackZhang-SH/TimeArchival3DGS.git

# Download Miniconda (Python 3.x)
wget -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Silent install to /workspace/miniconda3
bash miniconda.sh -b -p /workspace/miniconda3

# Enable `conda` (writes init to ~/.bashrc)
/workspace/miniconda3/bin/conda init bash

# Apply to current session
source ~/.bashrc
cd TimeArchival3DGS
conda create -n TimeArchival3DGS python=3.8
conda activate TimeArchival3DGS
pip install --upgrade pip

# PyTorch 2.2 + CUDA 12.1 wheels (most stable with Py3.8)
pip install --index-url https://download.pytorch.org/whl/cu121   --extra-index-url https://pypi.org/simple   torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2

# Optional debug patch (only if your build fails here)
sed -i '1i #include <cstdint>\n#include <stdint.h>'   submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h

# Common deps
pip install tqdm plyfile pillow imageio[ffmpeg] opencv-python joblib

# Build submodules (no isolation, no cache)
pip install --no-build-isolation --no-cache-dir ./submodules/diff-gaussian-rasterization
pip install --no-build-isolation --no-cache-dir ./submodules/simple-knn
pip install --no-build-isolation --no-cache-dir ./submodules/fused-ssim
```

---

## Dataset layout (TACV / time-indexed)

### Prepared example dataset (Google Drive)

Several prepared TA-3DGS datasets are available here:

- **Google Drive folder:** https://drive.google.com/drive/folders/1pPrEECcB7rmMX67Yfyd8Ok4sT4BbOjM4?usp=sharing

Download the folder and place the dataset under `./dataset/` (keep the original subfolder names, e.g., `frame_1/`, `frame_2/`, ...).

### Dataset note
The shared datasets are provided for **research and evaluation** purposes.  
Please ensure you have the rights to use any third-party assets contained in the data (e.g., captured footage / game renders).
**Please do not redistribute** third-party content without permission from the original rights holders.

**Folder naming convention (A vs B):**
- Folders with **`A`** in the name contain **static stadium/environment only (no players)** - intended for training a **high-quality static background 3DGS**.
- Folders with **`B`** in the name contain **stadium + players (dynamic scene)** - intended for **per-frame (time-indexed) training** where moving actors are present.

A typical TA-3DGS dataset root looks like:

```

dataset/<scene_name>/
  frame_1/
    images/
    sparse/0/            # COLMAP output (optional)
    fused_points.ply     # optional 3DGS init points (recommended)
  frame_2/
    images/
    sparse/0/
    fused_points.ply
  ...
  frame_N/
    ...

```

For a **single static scene** (no time indexing), you can also use the standard Gaussian-Splatting style layout and train with `train.py`.

---

## Train (stable)

### A) Train a single scene (non-time-indexed)
```bash
python train.py --source_path ./dataset/my_scene --model_path ./output/my_scene --disable_viewer
```

### B) Train all frames (time-indexed)
```bash
python ta_train.py -s ./dataset/soccer_B_70cams -o ./output_seq/soccer_B_70cams --frames all --resume-if-exists -- \
  --disable_viewer -r 1 --iterations 8000
```

Notes:
- `-s` points to the dataset root containing `frame_*`.
- `-o` is the output root; each frame will produce a model folder.
- The arguments after `--` are forwarded to the underlying trainer (iterations, resolution, etc.).

### C) Train all frames with warm start (recommended)
This uses `ta_train.py --warm_chain` to initialize each frame from the **previous frame’s checkpoint**.

```bash
python ta_train.py -s ./dataset/soccer_B_70cams -o ./output_seq/soccer_B_70cams --frames all --resume-if-exists --warm_chain -- \
  --disable_viewer -r 1 --iterations 8000 --densify_until_iter 0 --densification_interval 0
```


---

## Warm start (recommended)

TA-3DGS supports **warm-start training** for time-indexed sequences: initialize a frame's 3DGS from a previous solution (most commonly the **previous frame**). This typically improves convergence speed and stability, and in many sequences it makes **per-frame GT point clouds optional**.

### Do I need per-frame `fused_points.ply` (GT init)?
Many prepared TA-3DGS datasets include a per-frame `fused_points.ply` ("GT init"). However, in practice it is often **not strictly necessary**:

- If cameras are consistent and motion between adjacent frames is moderate, **Warm + NoDensify** is usually **close to GT init**.
- GT init tends to be only **slightly better** than Warm + NoDensify in many sequences, suggesting per-frame GT may be optional **except in harder motion regimes** (fast motion, heavy occlusion, large view changes, etc.).

### Quality vs cost: three practical modes

- **Warm + Densify**: Best quality, but cost grows rapidly over time (training slows, model size explodes).
- **Warm + NoDensify**: Stable and compact, quality close to GT but generally not better.
- **GT Init**: Stable baseline; quality steady and only slightly above Warm + NoDensify, suggesting per-frame GT may be optional except in harder motion regimes.

### How to disable densification
Warm start plus densification can become increasingly expensive on long sequences because the number of Gaussians keeps growing. For long sequences, **disable densification**.

To turn densification off, pass the following to the underlying trainer (arguments after `--`):

- `--densify_until_iter 0` (recommended)

Optionally, you can also set:

- `--densification_interval 0`

Example (time-indexed):
```bash
python ta_train.py -s ./dataset/soccer_B_70cams -o ./output_seq/soccer_B_70cams --frames all --resume-if-exists -- \
  --disable_viewer -r 1 --iterations 8000 --densify_until_iter 0 --densification_interval 0
```

Example (single scene):
```bash
python train.py --source_path ./dataset/my_scene --model_path ./output/my_scene --disable_viewer \
  --densify_until_iter 0 --densification_interval 0
```

### Enabling warm start

Warm start for time-indexed training is enabled by **`ta_train.py --warm_chain`**.

What it does:
- Trains frames **sequentially** (in increasing frame index).
- Forces a checkpoint at the **final iteration** by automatically appending `--checkpoint_iterations <N>`.
- Uses the previous frame’s checkpoint (`chkpnt<N>.pth`) as `--start_checkpoint` for the next frame **and** passes `--reset_start_iter` so the checkpoint is treated as initialization (iteration restarts from 0).

**Important:** `--warm_chain` requires that you pass `--iterations N` *after* the `--` (i.e., in the forwarded `train.py` args).

#### Example: Warm + NoDensify (recommended for long sequences)
```bash
python ta_train.py \
  -s ./dataset/soccer_B_70cams \
  -o ./output_seq/soccer_B_70cams \
  --frames all \
  --resume-if-exists \
  --warm_chain \
  -- \
  --disable_viewer -r 1 \
  --iterations 8000 \
  --densify_until_iter 0 --densification_interval 0
```

#### Example: Warm + Densify (best quality, but slows down over time)
```bash
python ta_train.py \
  -s ./dataset/soccer_B_70cams \
  -o ./output_seq/soccer_B_70cams \
  --frames all \
  --resume-if-exists \
  --warm_chain \
  -- \
  --disable_viewer -r 2 \
  --iterations 8000 \
  --densify_from_iter 1500 --densify_until_iter 5000 \
  --densification_interval 200 --densify_grad_threshold 5e-4 \
  --percent_dense 0.005 --opacity_reset_interval 6000
```

> Tip: If you also pass `--resume-if-exists`, warm-chaining will try to continue from an existing `chkpnt<N>.pth` inside each model folder. If the expected checkpoint is missing, that next frame will fall back to cold start for safety.

---

## Pack (stable, optional)

Packing converts trained outputs into `.pt` blobs for faster loading in the TA server.

### Pack a whole folder of per-frame models
```bash
python ta_pack.py -m ./output_seq/soccer_B_70cams --out ./output_seq_packed/soccer_B_70cams --prefix model_frame_ --autocreate
```

### Pack a single PLY (static reference) into a single `.pt`
```bash
python ta_pack.py --single_ply ./output_seq/soccer_A_70cams/point_cloud/iteration_30000/point_cloud.ply --out_pt ./output_seq_packed/static_A.pt
```

---

## TA server GUI (stable, optional)

Start the server and browse per-frame models:

```bash
python ta_server_slots.py \
  -p ./output_seq_packed/soccer_B_70cams \
  --prefix model_frame_ \
  --slots 3 \
  --host 127.0.0.1 \
  --port 7860 \
  --warmup \
  --neighbor_prefetch
```

Then open:
- `http://127.0.0.1:7860`

---

## Test / evaluate (stable)

Evaluate rendering quality per frame:

```bash
python ta_test.py \
  -s ./dataset/soccer_B_70cams \
  -m ./output_seq/soccer_B_70cams \
  --frames all \
  --prefix model_frame_ \
  --iteration -1 \
  --save_json ./output_seq/soccer_B_test_results.json
```

---

## Experimental: A+B merge for static stadium + dynamic actors (experimental)

This section describes an **experimental** workflow to render a merged result:
- **Scene A**: a high-quality static stadium/environment (no dynamic actors)
- **Scene B**: per-frame reconstructions that include dynamic actors
- Render-time merges A + B, to reduce memory and training cost.

### Input assumptions (strict)
This A+B merge workflow assumes:
1. **A and B have identical cameras**: intrinsics + extrinsics and **camera indices must match exactly**.
2. **A and B lighting must match** (same time-of-day / exposure assumptions).
3. **No occlusions / no significant visibility differences** between A and B for the background region.

### Known issues / limitations (current)
- The merge quality is sensitive to **lighting mismatch** and **background occlusions**.
- The current approach still reconstructs **full B frames** (but with fewer iterations), so the merged result is **not as clean** as fully independent, high-quality per-frame training.
- The big upside is **speed**: typically **~25% of the time** of "train-everything-independently" for the same sequence length (your mileage may vary).

### Current direction / roadmap
The long-term plan is to make merging robust and higher quality by:
1. **Precisely localizing each player**
2. Reconstructing only a **small region around players** as 3DGS
3. Using the stadium **mesh** (or a stable background representation) for the environment
4. Finally **fusing mesh + player 3DGS** into a consistent render-time pipeline

### Full pipeline scripts (A fine-train + B coarse + pack + merge + server + ta_test)
- Linux/macOS: `scripts/run_tacv_pipeline.sh`
- Windows PowerShell: `scripts/run_tacv_pipeline.ps1`

These scripts implement:
1) Fine-train A
2) Batch-train B (coarse)
3) Pack A and B
4) Precompute residual masks
5) Build filtered B-only (players)
6) Pack filtered B-only
7) Server renders **A+B merged on-the-fly**
8) `ta_test` evaluates the **merged** result

---

## Repository entry points

Common entry scripts:
- `train.py` - single scene training (non-time-indexed)
- `ta_train.py` - time-indexed per-frame training
- `ta_pack.py` - pack per-frame models (and/or a single PLY) into `.pt`
- `ta_server_slots.py` - GUI server for browsing / rendering per-frame models
- `ta_test.py` - per-frame rendering + evaluation
- `make_residual_masks.py` - (experimental) residual mask generation
- `merge_A_B_batch.py` - (experimental) A+B merge helper


---

## Credits and attribution

If you use this repository in academic work, please cite the project.  
For other usage, please follow the license terms and the third-party licenses of submodules.

**Maintainer:** Yunxiao (Jack) Zhang  
For questions, please open an issue or contact me.

### Related repositories
- **TACV (parent project / concept)**: **[Time-Archival Camera Virtualization for Visual Performance and Sports](https://github.com/JackZhang-SH/Time-Archival-Camera-Virtualization-for-Visual-Performance-and-Sports)**.
- **RS Studio (dataset exporter / tooling)**: **[RealSynth Studio](https://github.com/JackZhang-SH/RealSynth_Studio)**.

## License

This repository contains and depends on components derived from the original **gaussian-splatting** project.

- **Upstream (gaussian-splatting) license:** see `LICENSE.md` (**research/evaluation only; non-commercial**).
- **Third-party submodules:** see their respective licenses in `submodules/`.

When using, redistributing, or building on this repository as a whole (including upstream-derived code), the **gaussian-splatting license terms apply**.

For any use beyond research/evaluation (e.g., commercial/product use), please follow the upstream license and contact the original licensors if required.



### Suggested acknowledgement text
You can paste the following in your paper/repo README:

> This work uses TA-3DGS (Time Archival 3D Gaussian Splatting) by Yunxiao (Jack) Zhang, a time-indexed 3D Gaussian Splatting system for dynamic sports scenes built on top of the 3DGS ecosystem and TACV.

### BibTeX (repo-level citation)
Update fields as needed (year/version/URL):

```bibtex
@misc{zhang_ta3dgs,
  title        = {Time Archival 3D Gaussian Splatting (TA-3DGS)},
  author       = {Zhang, Yunxiao},
  howpublished = {\url{https://github.com/JackZhang-SH/TimeArchival3DGS}},
  year         = {2026}
}
```
