import gc
import threading

import numpy as np
import open3d as o3d
import pytest

from heaicare_capture_sim import renderer as renderer_mod
from heaicare_capture_sim.head import Head, make_synthetic_head
from heaicare_capture_sim.renderer import RenderWorker
from heaicare_capture_sim.rig import MAX_HEIGHT, MAX_WIDTH, Camera

COLORS = [
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
    (1, 0.5, 0),
    (0.5, 0, 1),
    (0, 0.5, 0.5),
]
MARKERS = np.array(
    [
        [x, 10.0 * (i % 3 - 1), z]
        for i, (x, z) in enumerate(
            [(x, z) for x in (-25.0, 0.0, 25.0) for z in (-25.0, 0.0, 25.0)]
        )
    ]
)
GATE_CAMERAS = [
    Camera(name="full", width=640, height=480, eye_mm=(0.0, -300.0, 0.0)),
    Camera(
        name="crop",
        width=320,
        height=240,
        focal_length_mm=4.0,
        eye_mm=(20.0, -300.0, 30.0),
    ),
]
TOLERANCE_PX = (
    0.27  # measured max 0.163 px (crop camera) + ~0.1 margin; must stay < 0.5
)


def _marker(center, color) -> Head:
    m = o3d.geometry.TriangleMesh.create_sphere(radius=2.0, resolution=40)
    m.translate(center)
    m.paint_uniform_color(color)
    return Head(mesh=m, texture=None, source="marker")


def _reprojection_errors(renderer, cam) -> np.ndarray:
    renderer.set_scene([_marker(p, c) for p, c in zip(MARKERS, COLORS, strict=True)])
    img = renderer.render(cam).astype(float)
    expected = cam.project(MARKERS)
    errors = []
    for color, uv in zip(COLORS, expected, strict=True):
        mask = np.linalg.norm(img - np.multiply(color, 255), axis=2) < 10
        v, u = np.nonzero(mask)
        assert len(u) > 50, f"marker {color} not found"
        errors.append(np.hypot(u.mean() - uv[0], v.mean() - uv[1]))
    return np.array(errors)


@pytest.mark.parametrize("cam", GATE_CAMERAS, ids=lambda c: c.name)
def test_reprojection_gate(renderer, cam):
    errors = _reprojection_errors(renderer, cam)
    print(f"{cam.name}: max {errors.max():.3f} px, mean {errors.mean():.3f} px")
    assert errors.max() < TOLERANCE_PX


def test_gate_catches_missing_pixel_corner_shift(renderer, monkeypatch):
    monkeypatch.setattr(renderer_mod, "_PIXEL_CORNER_SHIFT", 0.0)
    assert _reprojection_errors(renderer, GATE_CAMERAS[0]).min() > 0.5


def test_textured_head_is_visible(renderer):
    renderer.set_scene([make_synthetic_head()])
    img = renderer.render(
        Camera(name="c", width=640, height=480, eye_mm=(0.0, -300.0, 100.0))
    )
    assert img.shape == (480, 640, 3) and img.dtype == np.uint8
    assert (img != renderer.background_rgb).any(axis=2).mean() > 0.1
    assert img.std() > 5


def test_empty_scene_renders_background(renderer):
    renderer.set_scene([])
    img = renderer.render(
        Camera(name="c", width=64, height=48, eye_mm=(0.0, -300.0, 0.0))
    )
    assert (img == renderer.background_rgb).all()


def test_rendering_is_deterministic(renderer):
    cam = Camera(name="c", width=640, height=480, eye_mm=(0.0, -300.0, 100.0))
    images = []
    for _ in range(2):
        renderer.set_scene([make_synthetic_head()])
        images.append(renderer.render(cam))
    assert images[0].tobytes() == images[1].tobytes()


def test_camera_larger_than_renderer_is_rejected(renderer):
    big = Camera.model_construct(  # validation caps cameras at the renderer size
        name="big",
        width=MAX_WIDTH + 1,
        height=MAX_HEIGHT,
        eye_mm=(0.0, -300.0, 0.0),
        target_mm=(0.0, 0.0, 0.0),
        up=(0.0, 0.0, 1.0),
        focal_length_mm=8.0,
        pixel_size_um=3.45,
    )
    with pytest.raises(ValueError, match="larger than the renderer"):
        renderer.render(big)


def test_render_worker_uses_one_dedicated_thread():
    worker = RenderWorker(threading.get_ident)
    results = {
        worker.submit(lambda r: (r, threading.get_ident())).result() for _ in range(5)
    }
    worker.close()
    assert len(results) == 1
    ((created_on, ran_on),) = results
    assert created_on == ran_on != threading.get_ident()


def test_render_worker_releases_renderer_on_its_thread():
    released_on = []

    class Probe:
        def __del__(self):
            released_on.append(threading.get_ident())

    worker = RenderWorker(Probe)
    worker_thread = worker.submit(lambda _r: threading.get_ident()).result()
    worker.close()
    assert released_on == [worker_thread]


def test_failed_job_does_not_keep_renderer_alive_past_close():
    released_on = []

    class Probe:
        def __del__(self):
            released_on.append(threading.get_ident())

    def boom(_r):
        raise RuntimeError("boom")

    def read_result(future):
        try:
            future.result()
        except RuntimeError:
            pass

    gc.disable()
    try:
        worker = RenderWorker(Probe)
        worker_thread = worker.submit(lambda _r: threading.get_ident()).result()
        failed = worker.submit(boom)
        failed.add_done_callback(read_result)  # builds the exception/traceback cycle
        with pytest.raises(RuntimeError):
            failed.result()
        del failed
        worker.close()
        assert released_on == [worker_thread]
    finally:
        gc.enable()
