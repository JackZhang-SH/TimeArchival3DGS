import argparse, os, cv2, numpy as np

def imread_rgb8(p):
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    if im is None: return None
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def to_gray_diff(gt_rgb, a_rgb, blur_px=0):
    # L1 通道平均差
    diff = np.mean(np.abs(gt_rgb.astype(np.int16) - a_rgb.astype(np.int16)), axis=2).astype(np.float32)
    if blur_px > 0:
        k = int(blur_px) * 2 + 1
        diff = cv2.GaussianBlur(diff, (k, k), 0)
    return diff

def morph_bin(m, open_px=0, close_px=1, dilate_px=3):
    ker = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r*2+1, r*2+1))
    if open_px > 0:  m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  ker(open_px))
    if close_px > 0: m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker(close_px))
    if dilate_px > 0: m = cv2.dilate(m, ker(dilate_px), iterations=1)
    return m

def main():
    ap = argparse.ArgumentParser("Make residual masks from GT vs A-only renders")
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--a_dir", required=True, help="A-only renders (same filenames)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--gt_ext", default=".jpg")
    ap.add_argument("--a_ext", default=".png")
    ap.add_argument("--thr", type=float, default=20.0, help="RGB L1-avg threshold (0-255)")
    ap.add_argument("--blur_px", type=int, default=0)
    ap.add_argument("--open_px", type=int, default=0)
    ap.add_argument("--close_px", type=int, default=1)
    ap.add_argument("--dilate_px", type=int, default=5)
    # 可选：深度一致性
    ap.add_argument("--gt_depth_dir", default=None, help="npys with same basenames")
    ap.add_argument("--a_depth_dir", default=None, help="npys with same basenames")
    ap.add_argument("--depth_tol_m", type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    names = [f for f in os.listdir(args.gt_dir) if f.lower().endswith(args.gt_ext.lower())]
    names.sort()
    kept, total = 0, 0
    for fn in names:
        total += 1
        base = os.path.splitext(fn)[0]
        gt_p = os.path.join(args.gt_dir, base + args.gt_ext)
        a_p  = os.path.join(args.a_dir, base + args.a_ext)
        gt = imread_rgb8(gt_p); a  = imread_rgb8(a_p)
        if gt is None or a is None or gt.shape != a.shape:
            print(f"[skip] size mismatch or missing: {fn}")
            continue

        diff = to_gray_diff(gt, a, args.blur_px)
        m = (diff >= args.thr).astype(np.uint8) * 255

        # 可选：深度一致性，删除光照差引起的伪差异
        if args.gt_depth_dir and args.a_depth_dir:
            gdp = os.path.join(args.gt_depth_dir, base + ".npy")
            adp = os.path.join(args.a_depth_dir,  base + ".npy")
            if os.path.isfile(gdp) and os.path.isfile(adp):
                gd = np.load(gdp).astype(np.float32)
                ad = np.load(adp).astype(np.float32)
                ok = np.isfinite(gd) & np.isfinite(ad)
                depth_agree = np.zeros_like(m, dtype=bool)
                depth_agree[ok] = (np.abs(gd[ok] - ad[ok]) >= args.depth_tol_m)
                m = (m.astype(bool) & depth_agree).astype(np.uint8) * 255

        m = morph_bin(m, args.open_px, args.close_px, args.dilate_px)
        out_p = os.path.join(args.out_dir, base + ".png")
        cv2.imwrite(out_p, m)
        kept += 1

    print(f"[done] wrote {kept}/{total} masks to {args.out_dir}")

if __name__ == "__main__":
    main()
