# merge_A_B.py
# 简单 PLY 合并（ASCII/二进制都支持）；保持字段顺序与Graphdeco版一致
import struct, sys, os
from argparse import ArgumentParser

def read_ply_xyzcso(path):
    # 读取包含 xyz, f_dc, f_rest, opacity, scale, rot 的 PLY（Graphdeco格式）
    import plyfile, numpy as np
    ply = plyfile.PlyData.read(path)
    v = ply["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype("float32")
    op  = v["opacity"].astype("float32")[:,None]
    sc  = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1).astype("float32")
    rot = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype("float32")
    # SH/特征维度名可能因版本不同，这里以常见命名为例：
    f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype("float32")
    f_list = []
    k = 0
    while True:
        name = f"f_rest_{k}"
        if name in v.dtype.names:
            f_list.append(v[name].astype("float32")[:,None]); k += 1
        else:
            break
    f_rest = np.concatenate(f_list, axis=1) if f_list else np.zeros((xyz.shape[0],0), np.float32)
    return dict(xyz=xyz, opacity=op, scale=sc, rot=rot, f_dc=f_dc, f_rest=f_rest)

def write_ply_xyzcso(path, data):
    import plyfile, numpy as np
    xyz, op, sc, rot, f_dc, f_rest = [data[k] for k in ["xyz","opacity","scale","rot","f_dc","f_rest"]]
    n = xyz.shape[0]
    props = [
        ("x","f4"),("y","f4"),("z","f4"),
        ("f_dc_0","f4"),("f_dc_1","f4"),("f_dc_2","f4"),
    ]
    for i in range(f_rest.shape[1]):
        props.append((f"f_rest_{i}", "f4"))
    props += [
        ("opacity","f4"),
        ("scale_0","f4"),("scale_1","f4"),("scale_2","f4"),
        ("rot_0","f4"),("rot_1","f4"),("rot_2","f4"),("rot_3","f4"),
    ]
    import numpy as np
    arr = np.zeros(n, dtype=props)
    arr["x"]=xyz[:,0]; arr["y"]=xyz[:,1]; arr["z"]=xyz[:,2]
    arr["f_dc_0"]=f_dc[:,0]; arr["f_dc_1"]=f_dc[:,1]; arr["f_dc_2"]=f_dc[:,2]
    for i in range(f_rest.shape[1]):
        arr[f"f_rest_{i}"]=f_rest[:,i]
    arr["opacity"]=op[:,0]
    arr["scale_0"]=sc[:,0]; arr["scale_1"]=sc[:,1]; arr["scale_2"]=sc[:,2]
    arr["rot_0"]=rot[:,0]; arr["rot_1"]=rot[:,1]; arr["rot_2"]=rot[:,2]; arr["rot_3"]=rot[:,3]
    el = plyfile.PlyElement.describe(arr, "vertex")
    plyfile.PlyData([el], text=False).write(path)

if __name__ == "__main__":
    ap = ArgumentParser("Merge fixed A and trained B")
    ap.add_argument("--a_ply", required=True, help="第0帧A的point_cloud.ply（推荐选你最满意的迭代）")
    ap.add_argument("--b_ply", required=True, help="本帧训练得到的B的point_cloud.ply")
    ap.add_argument("--out_ply", required=True, help="输出合并后的PLY")
    args = ap.parse_args()
    A = read_ply_xyzcso(args.a_ply)
    B = read_ply_xyzcso(args.b_ply)
    import numpy as np
    OUT = { k: np.concatenate([A[k], B[k]], axis=0) for k in A.keys() }
    os.makedirs(os.path.dirname(args.out_ply), exist_ok=True)
    write_ply_xyzcso(args.out_ply, OUT)
    print(f"[merge] wrote {args.out_ply}  (A:{A['xyz'].shape[0]} + B:{B['xyz'].shape[0]})")
