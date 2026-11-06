# v1 plan and rendering spike

Status: revision 2 after review (2026-11-06). Nothing in `src/` has been implemented yet.

## Part A: Spike findings

Spike code: `spikes/01_offscreen_render/` (throwaway, not part of the package).

- `smoke.py`: minimal offscreen render.
- `reproj_check.py [seed] [halfpix]`: renders markers and a textured ellipsoid from a random camera with an
  OpenCV `K` (non-square pixels, principal point off-centre) and a world-to-camera `[R|t]` built with scipy.
  It then compares the result against `cv2.projectPoints`. Results and images go to `out/`.

Environment: Open3D 0.20.0 (CUDA wheel), numpy 2.5.4, OpenCV 5.0.0, scipy 1.18.1, Python 3.12.13.
Machine: Intel Lunar Lake laptop (Arc 130V/140V **integrated** GPU), Mesa 26.0.8.

### A1. API (verified by introspection of the installed 0.20.0)

| Need | API |
|---|---|
| Offscreen renderer | `o3d.visualization.rendering.OffscreenRenderer(width, height, resource_path='')` |
| Camera from K + extrinsic | `OffscreenRenderer.setup_camera(intrinsic_matrix: float64[3,3], extrinsic_matrix: float64[4,4], intrinsic_width_px: int, intrinsic_height_px: int)` (overload 3) |
| Explicit near/far | `ren.scene.camera.set_projection(intrinsics: float64[3,3], near_plane, far_plane, image_width, image_height)` (overload 3) |
| RGB | `render_to_image() -> geometry.Image` (uint8, H×W×3) |
| Depth | `render_to_depth_image(z_in_view_space: bool = False)`, float32, background = `inf` |
| Disable tone mapping / FXAA | `ren.scene.view.set_post_processing(False)` |

Docs: https://www.open3d.org/docs/release/python_api/open3d.visualization.rendering.OffscreenRenderer.html.
The docstrings quoted above come from `help()` on the installed build. The web docs may describe another release.

### A2. Headless rendering works as installed

- With `DISPLAY` and `WAYLAND_DISPLAY` unset, Open3D logs
  `GLFWWindowSystem: X11 GLFW init failed ... Falling back to headless mode.` and renders normally. No extra setup was needed.
- Filament picks the **Vulkan** backend on the Intel iGPU (Mesa). Note that CLAUDE.md calls this a laptop "without GPU". It has
  no discrete GPU, but it does have an integrated one, and the integrated GPU is being used.
- Software rendering also works. With `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json` (Mesa lavapipe), Open3D logs
  "Vulkan software device detected" and renders. This setup is the stand-in for a truly GPU-less host such as CI or Docker without a GPU.
- Images rendered headless and with a display are byte-identical.

### A3. Reprojection consistency with `cv2.projectPoints`

Setup: 1024×768 image, `fx=900, fy=880, cx=530.3, cy=370.7`, random rotation (`Rotation.random`), random camera centre,
40 white unlit spheres (r = 1.5 mm, 250–700 mm deep), and an intensity-weighted blob centroid compared against the projected sphere centre.

| Run | mean (du, dv) px | std (du, dv) px | max ‖err‖ px |
|---|---|---|---|
| seed 0, **K passed as-is** | **(−0.497, −0.501)** | (0.035, 0.035) | 0.774 |
| seed 0, K with cx, cy **+0.5** | (0.002, −0.006) | (0.034, 0.047) | 0.122 |
| seed 1, +0.5 | (0.004, 0.003) | (0.033, 0.033) | 0.089 |
| seed 2, +0.5 | (0.005, −0.003) | (0.034, 0.032) | 0.079 |
| seed 3, +0.5 | (0.001, −0.002) | (0.036, 0.035) | 0.119 |
| seed 0, +0.5, lavapipe | (0.008, −0.018) | (0.061, 0.070) | 0.247 |

Dense check: ellipsoid vertices were projected with OpenCV, and the predicted camera-z was compared with the rendered depth at the nearest pixel.
On front-facing vertices with the +0.5 correction, median |Δz| was 0.09–0.12 mm, p99 was 0.75–0.83 mm, and 93–96 % of vertices fell under 0.5 mm.
Most of the remaining error comes from rounding to the nearest pixel on sloped surfaces. As a negative control, the same check with the image y-axis flipped matched only 1–15 % of vertices.

### A4. Convention mismatches found

1. **Half-pixel principal-point offset (systematic −0.5 px on both axes).** Open3D/Filament treats `K` in GL pixel-corner coordinates,
   where pixel *i* spans [i, i+1). OpenCV centres pixel *i* on *i*. Fix at the backend boundary: hand Open3D `cx + 0.5, cy + 0.5`.
   Exported metadata keeps the user's OpenCV `K` unchanged. The reprojection gate must be able to detect this, because 0.5 px is 4× the worst error measured after correction.
2. **Axes.** `setup_camera(K, T)` accepts an OpenCV-axis world-to-camera matrix directly, with no flip needed.
   Internally, `camera.get_view_matrix()` equals `diag(1, −1, −1) @ [R|t]`, which is the OpenGL convention (verified numerically). It must never be exported.
3. **Depth semantics.** `render_to_depth_image(z_in_view_space=True)` returns camera **z** (median error 0.09 mm), not Euclidean range (median error 1.6 mm). Background pixels are `inf`.
4. **Near/far.** `setup_camera` sets near = 0.1 (unitless) and far from the scene bounding box (about 5.3 m here). Units are not forced.
   0.1 mm and the bbox-derived far plane worked. Overriding with `set_projection(K, 10, 5000, W, H)` left the results unchanged, so the backend will set near/far explicitly instead of relying on the bbox.
5. **COLMAP vs OpenCV.** These are *not* identical. COLMAP puts the centre of the top-left pixel at (0.5, 0.5), so `cx_colmap = cx_opencv + 0.5`.
   CLAUDE.md's "must match OpenCV/COLMAP" therefore needs a +0.5 shift in any COLMAP export. (This comes from COLMAP's documentation and has not been tested here.)

### A5. Other findings that affect the design

- **Only one `OffscreenRenderer` per process.** Two instances render correctly, but the process aborts at exit with SIGABRT (exit 134):
  `utils::PreconditionPanic ... destroying MaterialInstance "defaultLit" which is still in use`. This reproduced 3 out of 3 times, and `clear_geometry()` does not help.
  A single instance exits cleanly. So the capture uses one renderer and loops over cameras, and pytest needs a session-scoped renderer fixture.
- **Determinism.** On the same device, renders are byte-identical in-process, across processes and with or without a display (same SHA-256 for RGB and depth).
  On a different device (lavapipe vs Intel), images differ but stay within the tolerances above. Determinism is therefore per-device, and the manifest must record the device.
- **Camera inside a closed box** (the incubator case): the walls are fully visible from inside, whatever the triangle winding. Culling does not hide inner walls.
- **Lighting is not solved.** With `defaultLit`, the default lighting and post-processing off, the textured ellipsoid renders very dark (`out/ellipsoid_seed*.png`).
  Texture mapping itself works (`albedo_img` + `create_sphere(create_uv_map=True)`).
- **Speed.** A 180k-triangle sphere renders in 0.031 s per frame at 1024×768 and 0.068 s per frame at 2448×2048 (iGPU). Rendering will not be the bottleneck.

### A6. Ground-truth depth and mask by ray casting (follow-up check, revision 2)

- `o3d.t.geometry.RaycastingScene.create_rays_pinhole(K, extrinsic, w, h)` has the **same pixel-corner convention**. The ray for
  pixel (0, 0) passes through K-coordinate (0.5, 0.5), so the same `cx + 0.5, cy + 0.5` fix applies.
- Ray directions have z = 1 in camera frame, so `cast_rays(...)["t_hit"]` is camera **z** directly. (Checked on a plane at z = 400: `t_hit` is 400.0 at both the centre and corner pixels.)
- `cast_rays` also returns `geometry_ids` and `primitive_ids`, which give the per-pixel mesh and triangle needed for the vault mask.
- `cv2.imwrite` / `cv2.imread(..., IMREAD_UNCHANGED)` round-trips a float32 TIFF exactly.

## Part B: Plan (revision 2)

Changes from your review (2026-11-06):
- No SfM/MVS export. `cameras.json` stays OpenCV-only, and the COLMAP milestone is dropped.
- No incubator hood. The scene is the head in the middle of an open environment.
- Intrinsics are authored as focal length in mm plus pixel size.
- A ground-truth depth map and a cranial-vault mask are added per camera.

### B1. Repository layout

```
src/heaicare_capture_sim/
  __init__.py        main() -> cli
  cli.py             argparse; `capture` subcommand
  rig.py             pydantic Rig/Camera/Intrinsics/Pose; physical intrinsics -> K; look_at -> (R, t); validation
  scene.py           pydantic Scene (mesh entries, GT id, vault plane, lighting); path resolution; file hashing
  capture.py         orchestration: load -> per camera: RGB + depth + mask -> write output folder + manifest
  open3d_backend.py  the ONLY module importing open3d. RGB via OffscreenRenderer, depth + vault mask via
                     RaycastingScene. One OpenCV->Open3D conversion (+0.5 px, explicit near/far) shared by both.
  placeholders.py    `python -m heaicare_capture_sim.placeholders`: textured ellipsoid head + environment -> OBJ/MTL/PNG
tests/
  conftest.py        session-scoped renderer (one per process, see A5)
  test_rig.py        schema validation + convention round-trips + mm/um -> px conversion
  test_placeholders.py
  test_reprojection.py   PRIMARY GATE (RGB and ground-truth channels; reads the exported output folder)
  test_capture.py    end-to-end folder layout / manifest
docs/                this plan, conventions, determinism results
examples/            scene_open.json, rig_ring8.json (small, committed)
data/                gitignored: placeholder + third-party meshes
captures/            gitignored: outputs
spikes/              throwaway (see note at the end)
```

Backend boundary: `open3d_backend.py` exposes one function, roughly
`render(scene, camera) -> (rgb: uint8 HxWx3, depth_mm: float32 HxW, vault_mask: bool HxW)`.
Its inputs follow the OpenCV convention only. There is no ABC or registry until a second backend exists.

Ground truth comes from ray casting rather than from the rasterizer's depth buffer. Ray casting gives an exact geometric intersection at each pixel centre,
and it is independent of MSAA and depth-buffer precision. The rasterized depth is used only as a cross-check that RGB and ground truth are aligned.

### B2. Rig JSON (draft)

```json
{
  "schema_version": 1,
  "name": "ring8_r350",
  "cameras": [
    {
      "id": "cam00",
      "width_px": 2448,
      "height_px": 2048,
      "intrinsics": {
        "focal_length_mm": 12.0,
        "pixel_size_um": 3.45,
        "principal_point_offset_px": [0.0, 0.0]
      },
      "distortion": null,
      "pose": {"type": "look_at", "eye_mm": [350, 0, 200], "target_mm": [0, 0, 0], "up": [0, 0, 1]}
    },
    {
      "id": "cam01",
      "width_px": 2448,
      "height_px": 2048,
      "intrinsics": {"focal_length_mm": 12.0, "pixel_size_um": 3.45},
      "distortion": null,
      "pose": {"type": "extrinsic", "R": [[1,0,0],[0,1,0],[0,0,1]], "t_mm": [0, 0, 400]}
    }
  ]
}
```

- Intrinsics are physical:
  - `fx = fy = focal_length_mm / (pixel_size_um * 1e-3)`. For the example, 12 / 0.00345 = 3478.26 px.
  - `cx = (width_px - 1) / 2 + offset_x` and `cy = (height_px - 1) / 2 + offset_y`. That is the geometric sensor centre in the OpenCV pixel-centre convention.
  - The offset is optional and defaults to 0.
  - Sensor size is derived (`width_px * pixel_size_um`) rather than authored, so it cannot contradict the other values.
- Pixels are **square** (a single `pixel_size_um`). This is an assumption, and it holds for current machine-vision sensors.
  The backend and tests still take a general 3×3 `K`, so the reprojection sweep keeps testing fx ≠ fy.
- `pose` is a discriminated union.
  - `look_at` is converted to `R, t` with OpenCV axes (x right, y down, z forward), with `up` mapping to image-up.
  - `extrinsic` is world-to-camera, `X_cam = R @ X_world + t`. It is a 3×3 matrix (no quaternion ordering ambiguity), and the validator checks orthonormality and det = +1 (tolerance 1e-6).
- `distortion`: in v1 only `null` is accepted. Any other value is a validation error. The field exists so the schema can be extended later.
- Validation: unique ids, positive sizes, focal length and pixel size, principal point inside the image, `eye != target`, and `up` not parallel to the viewing direction.

### B3. Scene JSON (draft)

```json
{
  "schema_version": 1,
  "name": "open_placeholder",
  "ground_truth": "head",
  "meshes": [
    {
      "id": "head",
      "path": "../data/placeholder/head.obj",
      "R": [[1,0,0],[0,1,0],[0,0,1]],
      "t_mm": [0, 0, 0],
      "vault": {"plane_point_mm": [0, 0, 0], "plane_normal": [0, 0, 1]}
    },
    {"id": "environment", "path": "../data/placeholder/environment.obj"}
  ],
  "lighting": "TBD in M7"
}
```

- Paths are relative to the scene file. `R` and `t_mm` are optional mesh-to-world transforms (identity by default).
- World frame: origin at the head centre, z up, mm. The head sits in the middle of the environment.
- Environment (v1 placeholder): an open, textured surface under the head (mattress on a base). There are no walls and no hood.
- **Vault definition.** The `vault` field applies only to the ground-truth mesh. It is a plane in *mesh* coordinates, and triangles whose centroid lies on the normal side of the plane count as cranial vault.
  For the placeholder ellipsoid, the plane is horizontal through the centre. For real head models, the plane should come from anatomical landmarks (for example the nasion–inion plane).
  That choice is deferred until real models exist. If a single plane proves too crude, per-triangle labels can replace it later.

### B4. Output folder (one capture)

```
captures/<name>/
  images/cam00.png ...    lossless 8-bit RGB PNG, W×H as declared
  depth/cam00.tiff ...    float32 TIFF, camera z in mm (first hit, whole scene), 0 = no hit
  masks/cam00.png ...     8-bit PNG, 255 = visible cranial vault (first hit is a vault triangle of the GT mesh), 0 otherwise
  rig.json                byte-exact copy of input rig
  scene.json              byte-exact copy of input scene
  cameras.json            per camera: id, file names, width, height, focal_length_mm, pixel_size_um,
                          K (3×3), dist_coeffs [], R (3×3), t (mm), convention string
  manifest.json           see below
```

- The depth is **camera z**, not distance along the ray, so it can be back-projected directly with `K`, `R`, `t`.
- The mask covers only the *visible* vault. Parts hidden by the environment or by the head itself are excluded.

`manifest.json` contains:
- the UTC time and the argv;
- `ground_truth` {mesh id, resolved path, sha256, vault plane};
- the sha256 of every input file, including the MTL and textures;
- `software`: package version, git commit plus dirty flag, python, open3d, numpy, scipy, opencv, pydantic;
- `render`: backend, post-processing off, near/far, platform string, `VK_ICD_FILENAMES` if set.

### B5. CLI

```
heaicare-capture-sim capture --scene examples/scene_open.json --rig examples/rig_ring8.json --out captures/ring8_run1
```

The command refuses an existing non-empty `--out` and exits non-zero on validation errors.

### B6. Milestones

Each milestone is a small diff. Each one is "done" only when `uv run pytest` and `uv run ruff check .` pass.

| # | Milestone | Acceptance criterion |
|---|---|---|
| M0 | Hygiene | `.gitignore` adds `data/`, `captures/`, `spikes/*/out/`. The empty test suite passes. |
| M1 | `rig.py` schema + conventions | Each validation rule rejects its bad case, and non-null distortion is rejected. 12 mm / 3.45 µm gives fx = fy = 3478.26 px, and cx, cy = ((W−1)/2, (H−1)/2). `look_at`: `cv2.projectPoints(target)` lands on (cx, cy) within 1e-9 px, world-up maps to smaller v, and `eye` maps to camera origin. `R,t → JSON → R,t` is exact. `cv2.Rodrigues` agrees with `Rotation.from_matrix(R).as_rotvec()`. |
| M2 | Placeholder assets | `python -m heaicare_capture_sim.placeholders --out data/placeholder --seed 0` writes the head (textured ellipsoid) and the environment (textured mattress and base, no hood). Reloaded via Open3D: UVs and texture are present, the head bbox matches the semi-axes within 0.1 mm, and the same seed gives identical sha256. UV poles go on the ear axis, so the vault is not pinched. |
| M3 | `open3d_backend.py`: RGB | Single renderer per process with explicit near/far and +0.5 px conversion. A renderer created with `DISPLAY` and `WAYLAND_DISPLAY` unset renders a non-empty image of the declared size. |
| **M4** | **Reprojection gate (RGB)** | A measurement script sweeps N ≥ 20 random poses and intrinsics (general K: fx ≠ fy, off-centre, several resolutions) on the iGPU and on lavapipe. Mean-bias and max-error tolerances are set from that distribution and documented next to the test. The test passes. **Negative control:** with the +0.5 shift removed, the test must fail. |
| M5 | Ground-truth depth + vault mask | Ray cast via `RaycastingScene` with the same conversion. **Gate:** the M4 marker test is re-run on the ray-cast `geometry_ids` channel, with tolerances measured the same way and the same negative control. Depth at the pixels of projected head vertices equals the cv2-predicted z within a measured tolerance. Ray-cast depth agrees with rasterized depth (median and p99, measured), which shows the RGB, depth and mask are aligned. Every mask pixel back-projects (depth + K, R, t) to a point on the vault side of the plane, within a measured tolerance of the head surface. A vault occluded by a test box disappears from the mask. |
| M6 | `capture` end-to-end | A 3-camera example produces exactly the B4 layout. `cameras.json` equals the rig-derived values, and the manifest GT sha256 matches the file. **The M4/M5 gates are re-run reading only the files in the output folder.** |
| M7 | Lighting for SfM-grade texture | Lighting is described in `scene.json`, with directional lights preferred. SIFT keypoints inside the vault mask ≥ a threshold per camera, and saturated vault pixels ≤ a threshold. Both thresholds are measured, and the spike's dark render must fail. |
| M8 | Determinism report | Two runs in separate processes give byte-identical PNGs and TIFFs on this machine (test). The iGPU vs lavapipe spread is measured and documented in `docs/`. |

### B7. Risks, unknowns, assumptions

Risks and unknowns:
- **Pixel and axis conventions.** These are handled in one shared conversion function. They are guarded by the M4 and M5 gates, both with negative controls.
- **One renderer per process (A5).** Tests and the CLI must respect it. I have not verified whether `clear_geometry()` leaks memory over long runs.
- **Rasterized RGB vs ray-cast ground truth at silhouettes.** MSAA blends edge pixels, while ray casting samples the exact pixel centre. Mask boundaries will differ from the RGB edges by up to about 1 px. This is expected, and M5 will measure it.
- **Cross-device determinism.** Same-device output is bit-exact, cross-device output is not. The Windows RTX 4060 and the Docker/CUDA RTX 5090 are untested, and Docker needs a Vulkan ICD in the container.
- **Recording the GPU device name.** Filament prints it from C++ to stdout. The fallback is the platform string plus `VK_ICD_FILENAMES`.
- **Lighting and appearance (A5).** M7 exists for this.
- **OBJ texture loading.** I have not verified whether `read_triangle_mesh` or `read_triangle_model` keeps the MTL texture in 0.20.0. M2 will verify it.
- **Vault plane for real heads.** A single plane is a placeholder definition (B3).

Assumptions:
- Square pixels. The principal point is the sensor centre plus an optional offset.
- World frame: origin at the head centre, z up, mm.
- Placeholder head semi-axes are about 60 × 48 × 45 mm (term neonate). They are configurable.
- Ideal sensor: no noise, blur, exposure or rolling shutter. RGB is 8-bit.
- Depth background value is 0 (confirmed).
- MSAA is left at the Filament default. Post-processing is off.

Deliberately skipped (add when needed): lens distortion, rig generators (ring/dome), backend registry, COLMAP/SfM export, incubator hood, crib and exam-table environments.

### Resolved points

1. Depth background is 0 (confirmed 2026-11-06).
2. The placeholder vault plane is horizontal through the head centre (confirmed 2026-11-06).

### Progress

- M0 done (2026-11-06): `.gitignore` covers `data/`, `captures/` and `spikes/*/out/`. `tests/test_smoke.py` checks that the package imports and that `opencv-python` is not installed next to the headless build.
