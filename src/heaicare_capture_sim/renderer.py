"""Open3D offscreen rendering. The only module that imports open3d rendering."""

from __future__ import annotations

import gc
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor

import cv2
import numpy as np
import open3d as o3d
from open3d.visualization import rendering

from heaicare_capture_sim.head import Head
from heaicare_capture_sim.rig import MAX_HEIGHT, MAX_WIDTH, Camera
from heaicare_capture_sim.scenario import Scenario

# Requested grey; the rendered value is measured (see Renderer.background_rgb).
_BACKGROUND = (0.5, 0.5, 0.5)
NEAR_MM, FAR_MM = 1.0, 10_000.0
# Open3D's setup_camera/set_projection use pixel-corner coordinates; OpenCV K uses pixel centres.
_PIXEL_CORNER_SHIFT = 0.5


class Renderer:
    """Single OffscreenRenderer; cameras up to its size are rendered and cropped.

    Fixed neutral lighting: unlit materials, post-processing off. Not thread-safe,
    and must be destroyed on the thread that created it (see ``RenderWorker``).

    Attributes
    ----------
    background_rgb : tuple of int
        Background colour as it appears in rendered images (measured at startup,
        since Open3D darkens the requested value).
    """

    def __init__(self, width: int = MAX_WIDTH, height: int = MAX_HEIGHT) -> None:
        self.width, self.height = width, height
        self._r = rendering.OffscreenRenderer(width, height)
        self._r.scene.set_background([*_BACKGROUND, 1.0])
        self._r.scene.view.set_post_processing(False)
        empty = np.asarray(self._r.render_to_image())
        self.background_rgb: tuple[int, int, int] = tuple(
            int(c) for c in empty[0, 0, :3]
        )

    def set_scene(
        self, meshes: Sequence[Head], scenario: Scenario | None = None
    ) -> None:
        """Replace all geometry with ``meshes`` (unlit) and ``scenario`` (shaded)."""
        self._r.scene.clear_geometry()
        for i, (mesh, rgb) in enumerate(scenario.parts if scenario else ()):
            # Shaded by Open3D's fixed default light so walls and edges stay
            # visible; matte so no highlight moves with the camera.
            material = rendering.MaterialRecord()
            material.shader = "defaultLit"
            material.base_color = [*rgb, 1.0]
            material.base_roughness, material.base_reflectance = 1.0, 0.0
            self._r.scene.add_geometry(f"scenario{i}", mesh, material)
        for i, head in enumerate(meshes):
            material = rendering.MaterialRecord()
            material.shader = "defaultUnlit"
            mesh = head.mesh
            if head.texture is not None:
                if head.texture.shape[2] == 4:
                    # No unlit shader honours alpha. Reflectance off keeps the
                    # see-through areas within 2 levels of what lies behind.
                    material.shader = "defaultLitTransparency"
                    material.base_reflectance, material.base_roughness = 0.0, 1.0
                    mesh = o3d.geometry.TriangleMesh(mesh).compute_vertex_normals()
                # Open3D's OBJ reader stores textures with row 0 at v = 0, but
                # albedo_img samples row 0 at v = 1; flip so OBJ files render upright.
                material.albedo_img = o3d.geometry.Image(
                    np.ascontiguousarray(head.texture[::-1])
                )
            self._r.scene.add_geometry(f"mesh{i}", mesh, material)

    def render(self, cam: Camera) -> np.ndarray:
        """Render one camera, lens distortion included (no blur, gain or noise).

        Returns
        -------
        np.ndarray
            RGB image, shape (cam.height, cam.width, 3), uint8.
        """
        if not any(cam.dist):
            return self._render_pinhole(cam, cam.K, cam.width, cam.height)
        # Render the undistorted view of every output pixel, then resample it.
        ideal = cam.ideal_pixels()
        lo = np.floor(ideal.min(axis=(0, 1))).astype(int) - 1  # bilinear neighbours
        hi = np.ceil(ideal.max(axis=(0, 1))).astype(int) + 1
        k = cam.K.copy()
        k[:2, 2] -= lo  # ideal pixel lo becomes pixel 0 of the pinhole render
        w, h = hi - lo + 1
        pinhole = self._render_pinhole(cam, k, w, h)
        maps = (ideal - lo).astype(np.float32)
        return cv2.remap(pinhole, maps[..., 0], maps[..., 1], cv2.INTER_LINEAR)

    def _render_pinhole(
        self, cam: Camera, k: np.ndarray, width: int, height: int
    ) -> np.ndarray:
        """Render ``cam``'s pose through intrinsics ``k`` into a width x height image."""
        if width > self.width or height > self.height:
            raise ValueError(
                f"camera {cam.name!r} needs a {width}x{height} render (distortion "
                f"included), larger than the renderer ({self.width}x{self.height})"
            )
        k = k.copy()
        k[0, 2] += _PIXEL_CORNER_SHIFT
        k[1, 2] += _PIXEL_CORNER_SHIFT
        extrinsic = np.eye(4)
        extrinsic[:3, :3], extrinsic[:3, 3] = cam.R, cam.t_mm
        # Same K on a bigger viewport: pixel (u, v) maps identically, so the top-left crop is exact.
        self._r.setup_camera(k, extrinsic, self.width, self.height)
        self._r.scene.camera.set_projection(k, NEAR_MM, FAR_MM, self.width, self.height)
        image = np.asarray(self._r.render_to_image())
        return image[:height, :width, :3].copy()


class RenderWorker[R]:
    """Run every rendering job on one thread that also created the renderer."""

    def __init__(self, factory: Callable[[], R]) -> None:
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="render")
        self._renderer = self._pool.submit(factory).result()

    def submit[T](self, job: Callable[[R], T]) -> Future[T]:
        """Queue ``job(renderer)``; jobs run in submission order."""
        return self._pool.submit(job, self._renderer)

    def close(self) -> None:
        """Finish queued jobs, release the renderer on its own thread, stop the thread."""

        def release() -> None:
            # Open3D aborts if a renderer is destroyed off the thread that created it.
            self._renderer = None
            # A failed job's traceback holds the renderer; if a done-callback called
            # future.result() that is a reference cycle only the GC frees, possibly on
            # another thread. Collect it here, on the render thread.
            gc.collect()

        self._pool.submit(release).result()
        self._pool.shutdown(wait=True)
