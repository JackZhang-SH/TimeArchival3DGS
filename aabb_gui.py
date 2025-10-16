# aabb_gui.py
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np, json, argparse, os

def make_bbox_lines(aabb):
    # aabb: [xmin,xmax,ymin,ymax,zmin,zmax]
    xmin,xmax,ymin,ymax,zmin,zmax = aabb
    pts = np.array([
        [xmin,ymin,zmin],[xmax,ymin,zmin],[xmax,ymax,zmin],[xmin,ymax,zmin],
        [xmin,ymin,zmax],[xmax,ymin,zmax],[xmax,ymax,zmax],[xmin,ymax,zmax]
    ], dtype=np.float32)
    lines = np.array([[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]], np.int32)
    colors = np.tile(np.array([[0,1,0]], np.float32), (lines.shape[0],1))
    ls = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(pts),
                              lines=o3d.utility.Vector2iVector(lines))
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls

def clamp_pair(vmin, vmax, lo, hi):
    vmin = max(vmin, lo); vmax = min(vmax, hi)
    if vmin > vmax: vmin, vmax = vmax, vmin
    return vmin, vmax

def main(pc_path, init_json, out_json):
    app = gui.Application.instance
    app.initialize()

    # 读点云
    pcd = o3d.io.read_point_cloud(pc_path)
    if len(np.asarray(pcd.points)) == 0:
        raise SystemExit("Empty point cloud.")
    aabb0 = pcd.get_axis_aligned_bounding_box()
    lo = aabb0.get_min_bound(); hi = aabb0.get_max_bound()

    if init_json and os.path.exists(init_json):
        with open(init_json,'r') as f: j = json.load(f)
        aabb = [j["xmin"],j["xmax"],j["ymin"],j["ymax"],j["zmin"],j["zmax"]]
    else:
        aabb = [lo[0],hi[0], lo[1],hi[1], lo[2],hi[2]]

    # 窗口与场景
    w = gui.Application.instance.create_window("AABB Editor", 1280, 800)
    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(w.renderer)
    mat = rendering.MaterialRecord(); mat.shader = "defaultUnlit"; mat.point_size = 2.0
    scene.scene.add_geometry("pcd", pcd, mat)
    bbox = make_bbox_lines(aabb)
    scene.scene.add_geometry("bbox", bbox, rendering.MaterialRecord())
    scene.setup_camera(60.0, aabb0, aabb0.get_center())
    w.add_child(scene)

    # —— 原“右侧面板：6 滑杆 + 按钮”处，替换为 —— 
    em = w.theme.font_size
    panel = gui.Vert(0.25*em, gui.Margins(0.5*em,0.5*em,0.5*em,0.5*em))

    # 用 2 列网格，标签和滑条并排，压缩高度
    grid = gui.VGrid(2, 0.25*em)
    panel.add_child(grid)

    sliders = {}
    def add_slider(name, v, vmin, vmax):
        lbl = gui.Label(name)
        s = gui.Slider(gui.Slider.DOUBLE)
        s.set_limits(vmin, vmax); s.double_value = v
        sliders[name] = s
        grid.add_child(lbl); grid.add_child(s)
        return s

    sx0 = add_slider("xmin", aabb[0], lo[0], hi[0])
    sx1 = add_slider("xmax", aabb[1], lo[0], hi[0])
    sy0 = add_slider("ymin", aabb[2], lo[1], hi[1])
    sy1 = add_slider("ymax", aabb[3], lo[1], hi[1])
    sz0 = add_slider("zmin", aabb[4], lo[2], hi[2])
    sz1 = add_slider("zmax", aabb[5], lo[2], hi[2])

    def update_bbox():
        a = [sx0.double_value, sx1.double_value,
             sy0.double_value, sy1.double_value,
             sz0.double_value, sz1.double_value]
        a[0],a[1] = clamp_pair(a[0],a[1], lo[0],hi[0])
        a[2],a[3] = clamp_pair(a[2],a[3], lo[1],hi[1])
        a[4],a[5] = clamp_pair(a[4],a[5], lo[2],hi[2])
        scene.scene.remove_geometry("bbox")
        scene.scene.add_geometry("bbox", make_bbox_lines(a), rendering.MaterialRecord())
        return a

    for s in sliders.values():
        s.set_on_value_changed(lambda _ : update_bbox())

    
    def on_save(_):
        a = update_bbox()
        with open(out_json, "w") as f:
            json.dump({"xmin":a[0],"xmax":a[1],"ymin":a[2],"ymax":a[3],"zmin":a[4],"zmax":a[5]}, f, indent=2)
        print(f"[saved] {out_json}")
    def on_reset(_):
        sx0.double_value, sx1.double_value = lo[0], hi[0]
        sy0.double_value, sy1.double_value = lo[1], hi[1]
        sz0.double_value, sz1.double_value = lo[2], hi[2]
        update_bbox()

    toolbar = gui.Horiz()
    btn_save = gui.Button("Save AABB"); btn_save.set_on_clicked(on_save)
    btn_reset = gui.Button("Reset to PC bounds"); btn_reset.set_on_clicked(on_reset)
    toolbar.add_child(gui.Label(f"PC: {os.path.basename(pc_path)}"))
    toolbar.add_child(btn_save)
    toolbar.add_child(btn_reset)



    # 将面板放到右侧
    w.add_child(toolbar)
    w.add_child(panel)  # panel 仍然是上方，但在 toolbar 下方

    def on_layout(ctx):
        r = w.content_rect
        em = w.theme.font_size

        # 2) 工具栏固定高度，保证按钮永远可见
        toolbar_h = int(2.4 * em)  # 一行刚好；想更紧凑可改 2.2*em
        toolbar.frame = gui.Rect(r.x, r.y, r.width, toolbar_h)

        # 3) 面板高度按窗口大小取一个“相对安全值”
        #   - 最小 160，最多占窗口高度的 40%
        panel_h = max(160, int(r.height * 0.40))
        panel.frame = gui.Rect(r.x, r.y + toolbar_h, r.width, panel_h)

        # 4) 场景占下面剩余空间
        scene.frame = gui.Rect(r.x, r.y + toolbar_h + panel_h, r.width, r.height - toolbar_h - panel_h)

    w.set_on_layout(on_layout)

    app.run()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pc", required=True, help="要可视化的点云（PLY/PCD）")
    ap.add_argument("--init", default=None, help="可选：初始AABB json（含xmin...zmax）")
    ap.add_argument("--out", required=True, help="保存AABB的json路径")
    args = ap.parse_args()
    main(args.pc, args.init, args.out)
