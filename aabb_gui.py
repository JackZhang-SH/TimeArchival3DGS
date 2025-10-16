# aabb_gui.py
# 功能：
# - 顶部固定工具栏（含 Save / Reset），永不被裁切
# - AABB 线框使用线材质（unlitLine/line），修复 Filament “missing required attributes” 报错
# - 保存 JSON 时自动创建目录，并打印保存结果

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
import json, argparse, os, sys

def make_bbox_lines(aabb_vals):
    """aabb_vals: [xmin, xmax, ymin, ymax, zmin, zmax] -> LineSet"""
    xmin, xmax, ymin, ymax, zmin, zmax = aabb_vals
    pts = np.array([
        [xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin], [xmin, ymax, zmin],
        [xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax]
    ], dtype=np.float32)
    lines = np.array([
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7]
    ], dtype=np.int32)
    colors = np.tile(np.array([[0,1,0]], np.float32), (lines.shape[0], 1))  # 绿色
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(lines)
    )
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls

def clamp_pair(vmin, vmax, lo, hi):
    vmin = max(vmin, lo); vmax = min(vmax, hi)
    if vmin > vmax:
        vmin, vmax = vmax, vmin
    return vmin, vmax

def main(pc_path, init_json, out_json):
    app = gui.Application.instance
    app.initialize()

    # —— 读点云 & 初始 AABB ——
    pcd = o3d.io.read_point_cloud(pc_path)
    if len(np.asarray(pcd.points)) == 0:
        raise SystemExit("Empty point cloud.")

    aabb0 = pcd.get_axis_aligned_bounding_box()
    lo = aabb0.get_min_bound()
    hi = aabb0.get_max_bound()

    if init_json and os.path.exists(init_json):
        with open(init_json, 'r', encoding='utf-8') as f:
            j = json.load(f)
        aabb_vals = [j["xmin"], j["xmax"], j["ymin"], j["ymax"], j["zmin"], j["zmax"]]
    else:
        aabb_vals = [lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]]

    # —— 窗口与场景 ——
    w = gui.Application.instance.create_window("AABB Editor", 1280, 800)
    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(w.renderer)

    # 点云材质（点渲染）
    pc_mat = rendering.MaterialRecord()
    pc_mat.shader = "defaultUnlit"
    pc_mat.point_size = 2.0
    scene.scene.add_geometry("pcd", pcd, pc_mat)

    # 线框材质（避免 Filament 缺失属性报错）
    line_mat = rendering.MaterialRecord()
    # 常见可用的线着色器名称：unlitLine / line（随 Open3D 版本不同）
    # 先用 unlitLine，若你的版本不支持，可把下面改成 "line"
    line_mat.shader = "unlitLine"
    # 不同版本字段名可能不同，两个都设一下，兼容性更好（不存在时会被忽略）
    try:
        line_mat.line_width = 2.0
    except Exception:
        pass
    try:
        line_mat.thickness = 2.0
    except Exception:
        pass

    def set_bbox_lines(a):
        """重建/替换 AABB 线框（统一使用 line_mat）"""
        try:
            scene.scene.remove_geometry("bbox")
        except Exception:
            pass
        scene.scene.add_geometry("bbox", make_bbox_lines(a), line_mat)

    # 初始 AABB 线框
    set_bbox_lines(aabb_vals)

    # 相机
    scene.setup_camera(60.0, aabb0, aabb0.get_center())
    w.add_child(scene)

    # —— 顶部工具栏（按钮始终可见）& 上方参数面板 ——
    em = w.theme.font_size

    # 顶部工具栏
    toolbar = gui.Horiz()
    lbl_pc = gui.Label(f"PC: {os.path.basename(pc_path)}")
    btn_save = gui.Button("Save AABB")
    btn_reset = gui.Button("Reset to PC bounds")
    toolbar.add_child(lbl_pc)
    toolbar.add_child(btn_save)
    toolbar.add_child(btn_reset)
    w.add_child(toolbar)

    # 参数面板：用两列网格压缩高度（Label | Slider）
    panel = gui.Vert(0.25 * em, gui.Margins(0.5*em, 0.5*em, 0.5*em, 0.5*em))
    grid = gui.VGrid(2, 0.25 * em)
    panel.add_child(grid)

    sliders = {}
    def add_slider(name, v, vmin, vmax):
        lbl = gui.Label(name)
        s = gui.Slider(gui.Slider.DOUBLE)
        s.set_limits(vmin, vmax)
        s.double_value = v
        sliders[name] = s
        grid.add_child(lbl)
        grid.add_child(s)
        return s

    sx0 = add_slider("xmin", aabb_vals[0], lo[0], hi[0])
    sx1 = add_slider("xmax", aabb_vals[1], lo[0], hi[0])
    sy0 = add_slider("ymin", aabb_vals[2], lo[1], hi[1])
    sy1 = add_slider("ymax", aabb_vals[3], lo[1], hi[1])
    sz0 = add_slider("zmin", aabb_vals[4], lo[2], hi[2])
    sz1 = add_slider("zmax", aabb_vals[5], lo[2], hi[2])

    def update_bbox():
        a = [sx0.double_value, sx1.double_value,
             sy0.double_value, sy1.double_value,
             sz0.double_value, sz1.double_value]
        a[0], a[1] = clamp_pair(a[0], a[1], lo[0], hi[0])
        a[2], a[3] = clamp_pair(a[2], a[3], lo[1], hi[1])
        a[4], a[5] = clamp_pair(a[4], a[5], lo[2], hi[2])
        set_bbox_lines(a)
        return a

    for s in sliders.values():
        # 注意：避免 lambda 闭包陷阱，这里参数不用 s
        s.set_on_value_changed(lambda _v: update_bbox())

    # 保存与重置
    def on_save(_=None):
        a = update_bbox()
        try:
            out_abs = os.path.abspath(out_json)
            out_dir = os.path.dirname(out_abs)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)
            with open(out_abs, "w", encoding="utf-8") as f:
                json.dump({
                    "xmin": a[0], "xmax": a[1],
                    "ymin": a[2], "ymax": a[3],
                    "zmin": a[4], "zmax": a[5]
                }, f, indent=2)
            print(f"[saved] {out_abs}")
        except Exception as e:
            print(f"[save failed] {out_json} -> {e}", file=sys.stderr)

    def on_reset(_=None):
        sx0.double_value, sx1.double_value = lo[0], hi[0]
        sy0.double_value, sy1.double_value = lo[1], hi[1]
        sz0.double_value, sz1.double_value = lo[2], hi[2]
        update_bbox()

    btn_save.set_on_clicked(on_save)
    btn_reset.set_on_clicked(on_reset)

    # 可选：Ctrl+S 快捷保存（某些老版本不支持 set_on_key，则忽略）
    try:
        def on_key(e):
            if e.key == gui.KeyName.S and (e.modifiers & gui.KeyModifier.CTRL):
                on_save()
                return gui.EventCallbackResult.HANDLED
            return gui.EventCallbackResult.IGNORED
        w.set_on_key(on_key)
    except Exception:
        pass

    # 将参数面板放在工具栏下方
    w.add_child(panel)

    # —— 布局：工具栏固定高度；面板可调高度；场景占剩余 ——
    def on_layout(ctx):
        r = w.content_rect
        # 工具栏固定高度：一行按钮
        toolbar_h = int(2.4 * em)        # 紧凑可改 2.2*em
        toolbar.frame = gui.Rect(r.x, r.y, r.width, toolbar_h)

        # 面板高度：窗口的 40%，但不少于 160 像素（可按喜好调整）
        panel_h = max(160, int(r.height * 0.40))
        panel.frame = gui.Rect(r.x, r.y + toolbar_h, r.width, panel_h)

        # 场景：占用下面剩余空间
        scene.frame = gui.Rect(r.x, r.y + toolbar_h + panel_h,
                               r.width, r.height - toolbar_h - panel_h)

    w.set_on_layout(on_layout)

    app.run()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pc", required=True, help="要可视化的点云（PLY/PCD）")
    ap.add_argument("--init", default=None, help="可选：初始 AABB json（含 xmin…zmax）")
    ap.add_argument("--out", required=True, help="保存 AABB 的 json 路径")
    args = ap.parse_args()
    main(args.pc, args.init, args.out)
