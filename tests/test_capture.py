import json
import logging

import numpy as np
import open3d as o3d
import pytest

from heaicare_capture_sim.capture import apply_sensor, capture, new_capture_dir
from heaicare_capture_sim.head import Head, make_synthetic_head
from heaicare_capture_sim.rig import Camera, Rig, load_rig, ring

SMALL = ring(2, 300.0, 100.0, width=64, height=48)


class FakeRenderer:
    background_rgb = (55, 55, 55)

    def __init__(self, fail_on: str | None = None, rgb=(200, 100, 50)):
        self.fail_on, self.rgb = fail_on, rgb

    def set_scene(self, meshes, scenario=None):
        self.scene = list(meshes)

    def render(self, cam):
        if cam.name == self.fail_on:
            raise RuntimeError("boom")
        return np.full((cam.height, cam.width, 3), self.rgb, dtype=np.uint8)


def test_output_layout_and_cameras_json(tmp_path):
    out = capture(SMALL, make_synthetic_head(), tmp_path / "run", FakeRenderer())
    assert sorted(p.name for p in (out / "images").iterdir()) == [
        "cam00.png",
        "cam01.png",
    ]
    assert load_rig(out / "rig.json") == SMALL
    assert len(o3d.io.read_triangle_mesh(str(out / "ground_truth.ply")).triangles) > 0
    meta = json.loads((out / "cameras.json").read_text())
    assert meta["mesh_source"] == "synthetic" and meta["version"]
    assert meta["created"].endswith("Z")
    for entry, cam in zip(meta["cameras"], SMALL.cameras, strict=True):
        assert entry["image"] == f"images/{cam.name}.png"
        np.testing.assert_allclose(entry["K"], cam.K)
        assert entry["dist"] == [0.0] * 5
        np.testing.assert_allclose(entry["R"], cam.R)
        np.testing.assert_allclose(entry["t_mm"], cam.t_mm)


def test_extras_rendered_but_not_in_ground_truth(tmp_path):
    head = make_synthetic_head()
    extra = Head(
        o3d.geometry.TriangleMesh.create_box(5, 5, 5).translate((0, 70, 0)), None, "x"
    )
    head = Head(head.mesh, head.texture, head.source, extras=(extra,))
    renderer = FakeRenderer()
    out = capture(SMALL, head, tmp_path / "run", renderer)
    assert renderer.scene == [head, extra]
    gt = o3d.io.read_triangle_mesh(str(out / "ground_truth.ply"))
    assert gt.get_max_bound()[1] < 61  # the head ends at y = 60; the box is at 70


def test_non_empty_out_refused(tmp_path):
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "x").write_text("x")
    with pytest.raises(FileExistsError):
        capture(SMALL, make_synthetic_head(), tmp_path / "run", FakeRenderer())


def test_failure_leaves_no_partial_folder(tmp_path):
    with pytest.raises(RuntimeError):
        capture(
            SMALL,
            make_synthetic_head(),
            tmp_path / "run",
            FakeRenderer(fail_on="cam01"),
        )
    assert list(tmp_path.iterdir()) == []


def test_image_write_failure_raises_and_cleans_up(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "heaicare_capture_sim.capture.o3d.io.write_image", lambda *a, **k: False
    )
    with pytest.raises(OSError, match="could not write image"):
        capture(SMALL, make_synthetic_head(), tmp_path / "run", FakeRenderer())
    assert list(tmp_path.iterdir()) == []


def test_missing_parents_created(tmp_path):
    out = capture(
        SMALL, make_synthetic_head(), tmp_path / "a" / "b" / "run", FakeRenderer()
    )
    assert (out / "cameras.json").is_file()


def test_camera_seeing_nothing_warns(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        capture(
            SMALL,
            make_synthetic_head(),
            tmp_path / "run",
            FakeRenderer(rgb=FakeRenderer.background_rgb),
        )
    assert "cam00" in caplog.text and "sees no geometry" in caplog.text


def test_new_capture_dirs_are_unique(tmp_path):
    a, b = new_capture_dir(tmp_path), new_capture_dir(tmp_path)
    assert a != b and a.is_dir() and b.is_dir()


def test_real_render_end_to_end(renderer, tmp_path):
    rig = Rig(
        cameras=[Camera(name="c0", width=320, height=240, eye_mm=(0.0, -300.0, 100.0))]
    )
    out = capture(rig, make_synthetic_head(), tmp_path / "run", renderer)
    img = np.asarray(o3d.io.read_image(str(out / "images" / "c0.png")))
    assert img.shape == (240, 320, 3) and img.std() > 5


GREY = np.full((60, 80, 3), 128, np.uint8)


def test_ideal_sensor_returns_image_unchanged():
    assert apply_sensor(Camera(name="c", eye_mm=(0.0, -300.0, 0.0)), GREY) is GREY


def test_gain_scales_and_clips():
    c = Camera(name="c", eye_mm=(0.0, -300.0, 0.0), gain=1.5)
    image = np.array([[[100, 200, 0]]], np.uint8)
    assert apply_sensor(c, image).tolist() == [[[150, 255, 0]]]


def test_noise_has_requested_std_and_is_seeded_by_name():
    c = Camera(name="c", eye_mm=(0.0, -300.0, 0.0), noise_std=5.0)
    a, b = apply_sensor(c, GREY), apply_sensor(c, GREY)
    assert (a == b).all()
    assert np.std(a.astype(float) - 128) == pytest.approx(5.0, rel=0.05)
    other = apply_sensor(c.model_copy(update={"name": "d"}), GREY)
    assert (other != a).any()


def test_blur_spreads_a_bright_dot_evenly():
    image = np.zeros((41, 41, 3), np.uint8)
    image[20, 20] = 255
    out = apply_sensor(Camera(name="c", eye_mm=(0.0, -300.0, 0.0), blur_px=2.0), image)
    ring = [out[20, 23, 0], out[20, 17, 0], out[23, 20, 0], out[17, 20, 0]]
    assert out[20, 20, 0] < 255 and ring[0] > 0 and len(set(ring)) == 1


def test_capture_applies_sensor_and_exports_dist(tmp_path):
    dist = (-0.1, 0.0, 0.0, 0.0, 0.0)
    rig = ring(1, 300.0, 100.0, width=64, height=48, gain=0.5, dist=dist)
    out = capture(rig, make_synthetic_head(), tmp_path / "run", FakeRenderer())
    image = np.asarray(o3d.io.read_image(str(out / "images" / "cam00.png")))
    assert image[0, 0].tolist() == [100, 50, 25]
    meta = json.loads((out / "cameras.json").read_text())
    assert meta["cameras"][0]["dist"] == list(dist)
