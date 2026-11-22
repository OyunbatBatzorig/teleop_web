# Usage (needs pxr, e.g. ~/miniconda3/envs/grut/bin/python):
#   python -I export_wristcam_stl.py ~/VLA/unitree_sim_isaaclab/assets/robots/g1-29dof-dex1-base-fix-usd/g1_29dof_with_dex1_base_fix1.usd ../web/assets/dex1_wristcam.stl
# export the camera housing (mesh part above the Dex1 body) of left_hand_base_link as a binary STL in the Dex1 base frame
import sys, struct
import numpy as np
from pxr import Usd, UsdGeom
usd, out = sys.argv[1], sys.argv[2]
st = Usd.Stage.Open(usd); cache = UsdGeom.XformCache()
hb = st.GetPrimAtPath("/g1_29dof_with_hand_rev_1_0/left_hand_base_link"); inv = cache.GetLocalToWorldTransform(hb).GetInverse()
tris = []; meshes = []
for p in Usd.PrimRange(hb):
    if not p.IsA(UsdGeom.Mesh): continue
    m = UsdGeom.Mesh(p); mat = cache.GetLocalToWorldTransform(p) * inv
    pts = np.array([list(mat.Transform(v)) for v in m.GetPointsAttr().Get()])
    cnt = m.GetFaceVertexCountsAttr().Get(); idx = m.GetFaceVertexIndicesAttr().Get()
    meshes.append((str(p.GetPath()), len(pts), round(pts[:, 2].min(), 4), round(pts[:, 2].max(), 4)))
    k = 0
    for c in cnt:
        f = idx[k:k + c]; k += c
        for i in range(1, c - 1):
            t = pts[[f[0], f[i], f[i + 1]]]
            if t[:, 2].min() > 0.0325: tris.append(t)
for mm in meshes: print("mesh", mm)
tris = np.array(tris)
# hand_base (left) -> Dex1 base: hb x -> -y, hb y -> x, hb z -> z, no side offset (app adds +-0.003)
db = np.stack([tris[..., 1], -tris[..., 0], tris[..., 2]], -1).astype(np.float32)
print("triangles", len(db), "bbox", db.reshape(-1, 3).min(0).round(4), db.reshape(-1, 3).max(0).round(4))
with open(out, "wb") as f:
    f.write(b"dex1 wrist camera housing from unitree_sim_isaaclab USD".ljust(80, b" ")); f.write(struct.pack("<I", len(db)))
    for t in db:
        n = np.cross(t[1] - t[0], t[2] - t[0]); n = n / (np.linalg.norm(n) or 1)
        f.write(struct.pack("<12fH", *n, *t.reshape(-1), 0))
