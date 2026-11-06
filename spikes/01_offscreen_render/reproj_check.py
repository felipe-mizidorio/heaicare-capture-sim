"""Spike (throwaway): Open3D offscreen render from OpenCV K + world-to-camera [R|t].

Checks
------
1. Marker check: small unlit spheres at known 3D points; blob centroids in the
   rendered image vs ``cv2.projectPoints``.  Gives sub-pixel bias per axis.
2. Dense depth check: ellipsoid vertices projected with ``cv2.projectPoints``;
   predicted camera-z vs rendered depth (``z_in_view_space=True``) at that pixel.
3. near/far chosen by ``setup_camera`` and the effect of overriding them.
4. Camera inside a closed box (incubator case): are inner walls rendered?
5. In-process determinism (two renders, same renderer) + image hash for
   cross-process comparison.

Usage: uv run python spikes/01_offscreen_render/reproj_check.py [seed]
"""

import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as Rot

r = o3d.visualization.rendering
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

W, H = 1024, 768
# deliberately non-square pixels and off-centre principal point, to catch flips/offsets
K = np.array([[900.0, 0.0, 530.3], [0.0, 880.0, 370.7], [0.0, 0.0, 1.0]])
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
# HALFPIX=1: shift principal point by +0.5 px when handing K to Open3D
# (Open3D/GL pixel i spans [i, i+1); OpenCV pixel i is centred on i)
HALFPIX = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
K_o3d = K.copy()
K_o3d[:2, 2] += 0.5 * HALFPIX
rng = np.random.default_rng(SEED)

# --- pose: arbitrary rotation (scipy), camera centre somewhere in a 1 m cube (mm)
R_wc = Rot.random(random_state=SEED).as_matrix()  # world -> camera rotation
C = rng.uniform(-500, 500, 3)  # camera centre, world, mm
t_wc = -R_wc @ C
T_wc = np.eye(4)
T_wc[:3, :3], T_wc[:3, 3] = R_wc, t_wc
rvec, _ = cv2.Rodrigues(R_wc)


def cam_to_world(p_c: np.ndarray) -> np.ndarray:
    return (R_wc.T @ (p_c - t_wc).T).T


def project(p_w: np.ndarray) -> np.ndarray:
    uv, _ = cv2.projectPoints(p_w.reshape(-1, 1, 3), rvec, t_wc, K, None)
    return uv.reshape(-1, 2)


# --- markers: spread over the image, depths 250..700 mm, min 30 px apart
uv_m, z_m = [], []
while len(uv_m) < 40:
    uv = rng.uniform([40, 40], [W - 40, H - 40])
    if all(np.hypot(*(uv - q)) > 30 for q in uv_m):
        uv_m.append(uv)
        z_m.append(rng.uniform(250, 700))
uv_m, z_m = np.array(uv_m), np.array(z_m)
p_c = (np.linalg.inv(K) @ np.c_[uv_m, np.ones(len(uv_m))].T * z_m).T
markers_w = cam_to_world(p_c)

marker_mesh = o3d.geometry.TriangleMesh()
for p in markers_w:
    s = o3d.geometry.TriangleMesh.create_sphere(radius=1.5, resolution=30)
    s.translate(p)
    marker_mesh += s
marker_mesh.paint_uniform_color([1, 1, 1])

# --- textured ellipsoid ("neonatal head" placeholder), 400 mm in front of camera
ell = o3d.geometry.TriangleMesh.create_sphere(
    radius=1.0, resolution=150, create_uv_map=True
)
ell.vertices = o3d.utility.Vector3dVector(np.asarray(ell.vertices) * [60.0, 48.0, 45.0])
ell.rotate(Rot.random(random_state=SEED + 1).as_matrix(), center=(0, 0, 0))
ell.translate(cam_to_world(np.array([0.0, 0.0, 400.0])))
ell.compute_vertex_normals()
noise = sum(
    cv2.resize(
        rng.random((s, s)).astype(np.float32),
        (1024, 1024),
        interpolation=cv2.INTER_CUBIC,
    )
    / (i + 1)
    for i, s in enumerate([8, 32, 128, 512])
)
noise = (255 * (noise - noise.min()) / np.ptp(noise)).astype(np.uint8)
tex = np.dstack([noise, (0.8 * noise).astype(np.uint8), (0.6 * noise).astype(np.uint8)])

ren = r.OffscreenRenderer(W, H)
ren.scene.set_background([0, 0, 0, 1])
ren.scene.view.set_post_processing(False)  # no tone mapping / FXAA

unlit = r.MaterialRecord()
unlit.shader = "defaultUnlit"
lit = r.MaterialRecord()
lit.shader = "defaultLit"
lit.albedo_img = o3d.geometry.Image(np.ascontiguousarray(tex))

ren.scene.add_geometry("ellipsoid", ell, lit)
ren.scene.add_geometry("markers", marker_mesh, unlit)
ren.setup_camera(K_o3d, T_wc, W, H)
res: dict = {
    "seed": SEED,
    "halfpix_correction": HALFPIX,
    "near_far_auto": [ren.scene.camera.get_near(), ren.scene.camera.get_far()],
    "view_matrix": ren.scene.camera.get_view_matrix().tolist(),
}

# --- 1. marker check (ellipsoid hidden)
ren.scene.show_geometry("ellipsoid", False)


def marker_errors() -> dict:
    img = np.asarray(ren.render_to_image())
    gray = img[..., 0].astype(np.float64)
    n, lab = cv2.connectedComponents((gray > 10).astype(np.uint8))
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    cents = []
    for k in range(1, n):
        m = lab == k
        w = gray[m]
        cents.append([(uu[m] * w).sum() / w.sum(), (vv[m] * w).sum() / w.sum()])
    cents = np.array(cents)
    pred = project(markers_w)
    d = np.array(
        [cents[np.argmin(np.linalg.norm(cents - p, axis=1))] - p for p in pred]
    )
    return {
        "n_blobs": int(n - 1),
        "n_markers": len(pred),
        "mean_dxdy_px": d.mean(0).round(4).tolist(),
        "std_dxdy_px": d.std(0).round(4).tolist(),
        "max_err_px": float(np.linalg.norm(d, axis=1).max().round(4)),
        "unique_values_marker_px": sorted({int(v) for v in np.unique(gray[gray > 10])})[
            -3:
        ],
    }, img


res["markers_auto_nearfar"], img_m = marker_errors()
cv2.imwrite(str(OUT / f"markers_seed{SEED}.png"), img_m[..., ::-1])

# --- 3. override near/far explicitly and redo
ren.scene.camera.set_projection(K_o3d, 10.0, 5000.0, float(W), float(H))
res["near_far_override"] = [ren.scene.camera.get_near(), ren.scene.camera.get_far()]
res["markers_override_nearfar"], _ = marker_errors()

# --- 2. dense depth check on the ellipsoid (markers hidden)
ren.scene.show_geometry("markers", False)
ren.scene.show_geometry("ellipsoid", True)
rgb = np.asarray(ren.render_to_image())
depth = np.asarray(ren.render_to_depth_image(z_in_view_space=True))
cv2.imwrite(str(OUT / f"ellipsoid_seed{SEED}.png"), rgb[..., ::-1])
V = np.asarray(ell.vertices)
N = np.asarray(ell.vertex_normals)
Vc = (R_wc @ V.T).T + t_wc
uv = project(V)
facing = np.einsum("ij,ij->i", N, C - V) > 0.2 * np.linalg.norm(
    C - V, axis=1
)  # well front-facing
ui, vi = (
    np.round(uv).astype(int).T
)  # nearest pixel, assuming OpenCV pixel-centre convention
inb = facing & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
dz = depth[vi[inb], ui[inb]] - Vc[inb, 2]
rng_err = depth[vi[inb], ui[inb]] - np.linalg.norm(Vc[inb], axis=1)
res["dense_depth"] = {
    "depth_dtype": str(depth.dtype),
    "background_value": float(depth[0, 0]),
    "n_vertices_checked": int(inb.sum()),
    "median_abs_dz_mm": float(np.median(np.abs(dz))),
    "p99_abs_dz_mm": float(np.percentile(np.abs(dz), 99)),
    "frac_abs_dz_lt_0.5mm": float((np.abs(dz) < 0.5).mean()),
    "median_abs_range_err_mm (if depth were euclidean)": float(
        np.median(np.abs(rng_err))
    ),
}

# deliberately wrong convention for contrast: flip y in image (OpenGL-style)
dz_flip = depth[(H - 1 - vi[inb]), ui[inb]] - Vc[inb, 2]
res["dense_depth"]["frac_ok_if_y_flipped"] = float((np.abs(dz_flip) < 0.5).mean())

# --- 5. determinism (in-process)
rgb2 = np.asarray(ren.render_to_image())
res["determinism_in_process_max_absdiff"] = int(np.abs(rgb.astype(int) - rgb2).max())
res["rgb_sha256"] = hashlib.sha256(rgb.tobytes()).hexdigest()[:16]
res["depth_sha256"] = hashlib.sha256(depth.tobytes()).hexdigest()[:16]

# --- 4. camera inside a closed box (incubator walls seen from inside)
ren.scene.show_geometry("ellipsoid", False)
box = o3d.geometry.TriangleMesh.create_box(2000, 2000, 2000)
box.translate(C - 1000)
box.compute_vertex_normals()
box.paint_uniform_color([0, 1, 0])
ren.scene.add_geometry("box", box, unlit)
ren.setup_camera(K_o3d, T_wc, W, H)
inside = np.asarray(ren.render_to_image())
res["inside_box"] = {
    "near_far_auto": [ren.scene.camera.get_near(), ren.scene.camera.get_far()],
    "frac_green_px": float((inside[..., 1] > 100).mean()),
}
box.triangles = o3d.utility.Vector3iVector(
    np.asarray(box.triangles)[:, ::-1]
)  # flip winding
box.compute_vertex_normals()
ren.scene.remove_geometry("box")
ren.scene.add_geometry("box", box, unlit)
res["inside_box"]["frac_green_px_inward_normals"] = float(
    (np.asarray(ren.render_to_image())[..., 1] > 100).mean()
)

print(json.dumps(res, indent=1))
(OUT / f"result_seed{SEED}_hp{int(HALFPIX)}.json").write_text(json.dumps(res, indent=1))
