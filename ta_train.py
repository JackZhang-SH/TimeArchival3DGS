#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Archival 3DGS — Robust Batch Trainer (process isolation)
Trains N independent 3DGS models by spawning a fresh Python process per frame.

Examples:
  python ta_train.py -s ./dataset -o ./output_seq --frames 1-50 -- \
      --disable_viewer -r 2 --densify_from_iter 1500 --densify_until_iter 5000 \
      --densification_interval 200 --densify_grad_threshold 5e-4 \
      --percent_dense 0.005 --opacity_reset_interval 6000 --iterations 10000

Notes:
- Everything after the first literal "--" is forwarded verbatim to train.py.
- Each frame is trained in an isolated subprocess to avoid cross-frame state carryover.
"""

from __future__ import annotations
import argparse
import json
import os
import shlex
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple


# --------------------------- utils ---------------------------

def _ensure_flag(flags: List[str], name: str, takes_value: bool = False, default_value: str | None = None) -> List[str]:
    """
    Ensure a CLI flag exists in flags. If not, append it (and its value if required).
    """
    if name in flags:
        return flags
    if takes_value and default_value is not None:
        return flags + [name, str(default_value)]
    return flags + [name]


def _has_flag(flags: List[str], name: str) -> bool:
    try:
        idx = flags.index(name)
        return True
    except ValueError:
        return False


def parse_frames(frames_expr: str, dataset_root: Path) -> List[int]:
    """
    Accepts:
      - "all" (scan dataset_root/frame_*)
      - "even" or "odd" (based on discovered frame_* dirs)
      - "N-M"
      - "N,M,K"
      - "N"
      - "frame_N"
    """
    def scan_all() -> List[int]:
        found = []
        for p in sorted(dataset_root.glob("frame_*")):
            if p.is_dir():
                try:
                    found.append(int(p.name.split("_")[1]))
                except Exception:
                    pass
        return found

    s = frames_expr.strip().lower()
    if s == "all":
        return scan_all()
    if s == "even" or s == "odd":
        all_frames = scan_all()
        if s == "even":
            return [i for i in all_frames if i % 2 == 0]
        else:
            return [i for i in all_frames if i % 2 == 1]
    if s.startswith("frame_"):
        return [int(s.split("_")[1])]
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    if "," in s:
        return [int(x) for x in s.split(",")]
    return [int(s)]


@dataclass
class FrameResult:
    frame: int
    ok: bool
    seconds: float
    model_dir: str
    returncode: int
    error: str | None = None
    cmd: str | None = None


# --------------------------- main ---------------------------

def main(argv: List[str]) -> None:
    wrapper = argparse.ArgumentParser(
        description="Time Archival 3DGS — Robust Batch Trainer (process isolation)",
        add_help=True
    )
    wrapper.add_argument("-s", "--source_root", type=str, required=True,
                         help="Root containing frame_1..frame_N COLMAP datasets")
    wrapper.add_argument("-o", "--output_root", type=str, required=True,
                         help="Root to write model_frame_* subfolders")
    wrapper.add_argument("--frames", type=str, default="all",
                         help='Range/list like "all" | "1-10" | "1,3,5" | "12" | "even" | "odd" | "frame_7"')
    wrapper.add_argument("--prefix", type=str, default="model_frame_",
                         help='Output subfolder name prefix (default: "model_frame_")')
    wrapper.add_argument("--per-frame-subdir", type=str, default=None,
                         help="Optional subdir under each model (e.g., experiment tag)")
    wrapper.add_argument("--resume-if-exists", action="store_true",
                         help="If target model dir already exists & seems trained, skip this frame.")
    wrapper.add_argument("--dry-run", action="store_true", help="Print commands only, do not execute.")
    wrapper.add_argument("--jobs", type=int, default=1,
                         help="Reserved for future parallelism (currently sequential).")
    # sentinel to split args
    wrapper.add_argument("--", dest="dashdash", action="store_true", help=argparse.SUPPRESS)

    # Split wrapper vs train flags
    if "--" in argv:
        dd = argv.index("--")
        wrapper_args = argv[:dd]
        train_flags = argv[dd + 1:]
    else:
        wrapper_args = argv
        train_flags = []

    args = wrapper.parse_args(wrapper_args)

    src_root = Path(args.source_root).resolve()
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    frames = parse_frames(args.frames, src_root)
    print(f"[TA-Train] Frames to train: {frames}")

    # Always disable viewer unless user explicitly passed it
    if not _has_flag(train_flags, "--disable_viewer"):
        train_flags = _ensure_flag(train_flags, "--disable_viewer", takes_value=False)

    results: List[FrameResult] = []

    for i in frames:
        ds_dir = src_root / f"frame_{i}"
        if not ds_dir.exists():
            print(f"[TA-Train] SKIP frame_{i}: dataset not found -> {ds_dir}")
            results.append(FrameResult(i, False, 0.0, "", 2, "dataset_missing"))
            continue

        # Compose model_dir
        model_dir = out_root / f"{args.prefix}{i}"
        if args.per_frame_subdir:
            model_dir = model_dir / args.per_frame_subdir
        model_dir.mkdir(parents=True, exist_ok=True)

        # Resume policy: if a checkpoint or trained artifact already exists, skip
        if args.resume_if_exists:
            ckpt_dir = model_dir / "chkpnt"
            gauss_ply = model_dir / "point_cloud" / "iteration_0" / "point_cloud.ply"
            maybe_trained = ckpt_dir.exists() or gauss_ply.exists()
            if maybe_trained:
                print(f"[TA-Train] RESUME-SKIP frame_{i}: {model_dir}")
                results.append(FrameResult(i, True, 0.0, str(model_dir), 0, None, cmd=None))
                continue

        # Build command for subprocess
        cmd = [sys.executable, "-u", "train.py",  # -u: unbuffered for real-time logs
               "--source_path", str(ds_dir),
               "--model_path", str(model_dir)] + train_flags

        human_cmd = " ".join(shlex.quote(x) for x in cmd)
        print(f"[TA-Train][spawn] frame_{i} -> {human_cmd}")

        if args.dry_run:
            results.append(FrameResult(i, True, 0.0, str(model_dir), 0, None, human_cmd))
            continue

        t0 = time.time()
        # Use a *fresh* environment per frame to avoid leaking variables between runs.
        env = os.environ.copy()
        # Make runs reproducible-ish without touching the training code:
        # At least fix Python hashing; user can still pass seeds to train.py if supported.
        env.setdefault("PYTHONHASHSEED", "0")
        # (Avoid inheriting any accidental experiment flags via env variables)
        for k in ["TA_EXPERIMENT", "TA_EVAL", "TA_TRAIN_TEST_EXP"]:
            env.pop(k, None)

        try:
            # Run training in an isolated Python process
            import subprocess
            proc = subprocess.run(cmd, env=env, cwd=str(Path(__file__).resolve().parent),
                                  check=False)  # don't raise; we record status below
            rc = proc.returncode
        except Exception as e:
            dt = time.time() - t0
            print(f"[TA-Train][ERROR] frame_{i}: spawn failed -> {e}")
            results.append(FrameResult(i, False, dt, str(model_dir), 3, str(e), human_cmd))
            continue

        dt = time.time() - t0
        ok = (rc == 0)
        status = "OK" if ok else f"FAIL(rc={rc})"
        print(f"[TA-Train][done] frame_{i}: {status} in {dt:.2f}s  -> {model_dir}")

        results.append(FrameResult(i, ok, dt, str(model_dir), rc, None if ok else "train_failed", human_cmd))

        # No need to manually clear CUDA here; subprocess exit releases everything.

    # Write a compact summary for bookkeeping
    summary = {
        "source_root": str(src_root),
        "output_root": str(out_root),
        "frames": frames,
        "train_flags": train_flags,
        "results": [asdict(r) for r in results],
        "ok_frames": [r.frame for r in results if r.ok],
        "failed_frames": [r.frame for r in results if not r.ok],
        "time_total_sec": sum(r.seconds for r in results),
    }
    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    n_fail = len([r for r in results if not r.ok])
    if n_fail:
        print(f"[TA-Train] Completed with failures at frames: "
              f"{[r.frame for r in results if not r.ok]}")
        sys.exit(1)
    else:
        print("[TA-Train] All frames trained successfully.")


if __name__ == "__main__":
    main(sys.argv[1:])
