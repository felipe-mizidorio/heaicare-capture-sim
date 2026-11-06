"""Spike: minimal offscreen render smoke test (throwaway)."""

import sys

import numpy as np
import open3d as o3d

r = o3d.visualization.rendering
ren = r.OffscreenRenderer(320, 240)
mat = r.MaterialRecord()
mat.shader = "defaultUnlit"
m = o3d.geometry.TriangleMesh.create_box(1, 1, 1)
m.paint_uniform_color([1, 0, 0])
ren.scene.add_geometry("box", m, mat)
ren.setup_camera(60.0, [0.5, 0.5, 0.5], [3, 3, 3], [0, 0, 1])
img = np.asarray(ren.render_to_image())
print("shape", img.shape, img.dtype, "red px", int((img[..., 0] > 200).sum()))
o3d.io.write_image(
    sys.argv[1] if len(sys.argv) > 1 else "smoke.png", o3d.geometry.Image(img)
)
