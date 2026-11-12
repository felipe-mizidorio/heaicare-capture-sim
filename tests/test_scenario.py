import json

import numpy as np
import open3d as o3d
import pytest

from heaicare_capture_sim.capture import capture
from heaicare_capture_sim.cli import main
from heaicare_capture_sim.gui import _scenario_options
from heaicare_capture_sim.head import make_synthetic_head
from heaicare_capture_sim.rig import Camera, ring, save_rig
from heaicare_capture_sim.scenario import ScenarioSpec, load_scenario, save_scenario

GREYBOX = "scenarios/greybox/scenario.json"


def _floor_scenario(folder, **spec) -> str:
    """A 2 x 2 m grey floor slab in a y-up cm model, 20 cm below the head."""
    folder.mkdir(parents=True, exist_ok=True)
    slab = o3d.geometry.TriangleMesh.create_box(200.0, 1.0, 200.0).translate(
        (-100.0, -1.0, -100.0)
    )
    assert o3d.io.write_triangle_mesh(str(folder / "floor.obj"), slab)
    fields = {"file": "floor.obj", "units": "cm", "up": "+y", "front": "+z"}
    fields |= {"head": [0.0, 20.0, 0.0], **spec}
    (folder / "scenario.json").write_text(json.dumps(fields))
    return str(folder / "scenario.json")


def test_spec_maps_head_to_origin_and_up_to_z():
    spec = ScenarioSpec(file="m.fbx", units="cm", up="+y", front="+z", head=(1, 2, 3))
    t = spec.to_world()
    np.testing.assert_allclose(t @ [1, 2, 3, 1], [0, 0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(t @ [1, 3, 3, 1], [0, 0, 10, 1], atol=1e-12)  # up
    np.testing.assert_allclose(t @ [1, 2, 4, 1], [0, 10, 0, 1], atol=1e-12)  # front
    assert np.linalg.det(t[:3, :3]) == pytest.approx(1000.0)  # no mirror


@pytest.mark.parametrize(
    "bad",
    [
        {"units": "furlong"},
        {"up": "y"},
        {"front": "+y"},  # parallel to up
        {"file": "../floor.obj"},
        {"head": [0, float("nan"), 0]},
        {"baby_yaw_deg": float("inf")},
        {"extra": 1},
    ],
)
def test_bad_scenario_json_rejected(tmp_path, bad):
    path = _floor_scenario(tmp_path / "s")
    data = json.loads((tmp_path / "s/scenario.json").read_text()) | bad
    (tmp_path / "s/scenario.json").write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_scenario(path)


def test_load_scenario_names_and_errors(tmp_path, monkeypatch):
    assert load_scenario("none") is None
    with pytest.raises(FileNotFoundError):
        load_scenario(tmp_path / "missing/scenario.json")
    monkeypatch.chdir(tmp_path)
    _floor_scenario(tmp_path / "scenarios/floor")
    assert load_scenario("floor").source == "scenarios/floor/scenario.json"
    (tmp_path / "scenarios/floor/floor.obj").unlink()
    with pytest.raises(FileNotFoundError, match="model"):
        load_scenario("floor")


def test_floor_lands_below_the_head(tmp_path):
    (mesh, rgb), *_ = load_scenario(_floor_scenario(tmp_path / "s")).parts
    lo, hi = mesh.get_min_bound(), mesh.get_max_bound()
    np.testing.assert_allclose(hi[2], -200.0, atol=1e-6)  # 20 cm below, in mm
    np.testing.assert_allclose([lo[0], hi[0]], [-1000.0, 1000.0], atol=1e-6)
    assert len(rgb) == 3


def test_baby_pose_moves_the_room_the_other_way():
    spec = ScenarioSpec(
        file="m.fbx", units="cm", up="+z", front="+y", head=(0, 0, 0),
        baby_yaw_deg=90.0, baby_offset_mm=(5.0, 0.0, 0.0),
    )  # fmt: skip
    t = spec.to_world()
    # The room point the baby moved to (5 mm = 0.5 cm along x) is the head centre.
    np.testing.assert_allclose(t @ [0.5, 0, 0, 1], [0, 0, 0, 1], atol=1e-12)
    # Turned 90° left, the baby faces room -x: 1 cm further along -x is +y (front).
    np.testing.assert_allclose(t @ [-0.5, 0, 0, 1], [0, 10, 0, 1], atol=1e-12)


def test_moved_room_matches_reload_after_save(tmp_path):
    path = _floor_scenario(tmp_path / "s")
    original = (tmp_path / "s/scenario.json").read_text()
    scenario = load_scenario(path)
    save_scenario(scenario)  # unmoved: no baby pose fields written
    assert json.loads((tmp_path / "s/scenario.json").read_text()) == json.loads(
        original
    )
    moved = scenario.with_baby((10, -20, 30), 33.0).with_baby((1.0, 2.0, 3.0), 45.0)
    assert scenario.spec.baby_offset_mm == (0.0, 0.0, 0.0)  # original untouched
    with pytest.raises(ValueError):
        scenario.with_baby((float("nan"), 0.0, 0.0), 0.0)
    save_scenario(moved)
    reloaded = load_scenario(path)
    assert reloaded.spec == moved.spec
    for (a, _), (b, _) in zip(moved.parts, reloaded.parts, strict=True):
        np.testing.assert_allclose(
            np.asarray(a.vertices), np.asarray(b.vertices), atol=1e-9
        )


def test_greybox_loads():
    scenario = load_scenario(GREYBOX)
    floor_z = min(m.get_min_bound()[2] for m, _ in scenario.parts)
    assert -1500 < floor_z < -1000  # head ~1.4 m above the floor


def test_scenario_is_rendered_without_touching_head_pixels(renderer, tmp_path):
    head = make_synthetic_head()
    cam = Camera(
        name="c", width=320, height=240, focal_length_mm=1.0, eye_mm=(0, -300, 100)
    )
    renderer.set_scene([head])
    alone = renderer.render(cam)
    renderer.set_scene([head], load_scenario(_floor_scenario(tmp_path / "s")))
    with_floor = renderer.render(cam)
    on_head = (alone != renderer.background_rgb).any(axis=2)
    assert on_head.any() and (with_floor[~on_head] != alone[~on_head]).any()
    # Silhouette pixels are anti-aliased against what lies behind; inside
    # them the unlit head is untouched by the scenario's light.
    inside = on_head.copy()
    for axis in (0, 1):
        for shift in (1, -1):
            inside &= np.roll(on_head, shift, axis)
    np.testing.assert_array_equal(with_floor[inside], alone[inside])


class _Recorder:
    background_rgb = (55, 55, 55)

    def set_scene(self, meshes, scenario=None):
        self.scenario = scenario

    def render(self, cam):
        return np.full((cam.height, cam.width, 3), 200, np.uint8)


def test_capture_records_scenario_and_keeps_it_out_of_ground_truth(tmp_path):
    rig, head = ring(2, 300.0, 100.0, width=64, height=48), make_synthetic_head()
    scenario = load_scenario(_floor_scenario(tmp_path / "s"))
    plain = capture(rig, head, tmp_path / "plain", _Recorder())
    renderer = _Recorder()
    out = capture(rig, head, tmp_path / "room", renderer, scenario)
    assert renderer.scenario is scenario
    meta = json.loads((out / "cameras.json").read_text())
    assert meta["scenario_source"] == scenario.source
    assert ScenarioSpec.model_validate(meta["scenario"]) == scenario.spec
    assert json.loads((plain / "cameras.json").read_text())["scenario_source"] is None
    gt = (out / "ground_truth.ply").read_bytes()
    assert gt == (plain / "ground_truth.ply").read_bytes()


def test_cli_bad_scenario_exits_2(tmp_path, capsys):
    save_rig(ring(2, 300.0, 100.0), tmp_path / "rig.json")
    args = [
        "capture",
        "--rig",
        str(tmp_path / "rig.json"),
        "--out",
        str(tmp_path / "o"),
    ]
    assert main([*args, "--scenario", str(tmp_path / "nope.json")]) == 2
    assert "scenario not found" in capsys.readouterr().err


def test_scenario_options_list_scenario_folders(tmp_path):
    for name in ("b/scenario.json", "a/scenario.json", "c/model.fbx"):
        (tmp_path / name).parent.mkdir(exist_ok=True)
        (tmp_path / name).write_text("")
    found = [str(tmp_path / "a/scenario.json"), str(tmp_path / "b/scenario.json")]
    assert _scenario_options(tmp_path, "none") == ["none", *found]
    assert _scenario_options(tmp_path, "/x/scenario.json") == [
        "none",
        *found,
        "/x/scenario.json",
    ]
