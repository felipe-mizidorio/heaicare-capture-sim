import json
import logging

import numpy as np
import open3d as o3d
import pytest

from heaicare_capture_sim.capture import capture, new_capture_dir
from heaicare_capture_sim.head import make_synthetic_head
from heaicare_capture_sim.rig import Camera, Rig, load_rig, ring

SMALL = ring(2, 300.0, 100.0, width=64, height=48)


class FakeRenderer:
    background_rgb = (55, 55, 55)

    def __init__(self, fail_on: str | None = None, rgb=(200, 100, 50)):
        self.fail_on, self.rgb = fail_on, rgb

    def set_scene(self, meshes):
        pass

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
    for entry, cam in zip(meta["cameras"], SMALL.cameras, strict=True):
        assert entry["image"] == f"images/{cam.name}.png"
        np.testing.assert_allclose(entry["K"], cam.K)
        np.testing.assert_allclose(entry["R"], cam.R)
        np.testing.assert_allclose(entry["t_mm"], cam.t_mm)


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


def test_missing_parents_created(tmp_path):  # Review Focus 2
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


def test_new_capture_dirs_are_unique(tmp_path):  # Review Focus 4
    a, b = new_capture_dir(tmp_path), new_capture_dir(tmp_path)
    assert a != b and a.is_dir() and b.is_dir()


def test_real_render_end_to_end(renderer, tmp_path):
    rig = Rig(
        cameras=[Camera(name="c0", width=320, height=240, eye_mm=(0.0, -300.0, 100.0))]
    )
    out = capture(rig, make_synthetic_head(), tmp_path / "run", renderer)
    img = np.asarray(o3d.io.read_image(str(out / "images" / "c0.png")))
    assert img.shape == (240, 320, 3) and img.std() > 5
