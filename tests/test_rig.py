import math

import numpy as np
import pytest
from pydantic import ValidationError

from heaicare_capture_sim.rig import Camera, Rig, dome, load_rig, ring, save_rig


def cam(**kw) -> Camera:
    return Camera(**{"name": "c", "eye_mm": (0.0, -300.0, 0.0), **kw})


def test_intrinsics_from_datasheet_values():
    k = cam(width=1920, height=1080, focal_length_mm=8.0, pixel_size_um=3.45).K
    f = 8.0 * 1000 / 3.45
    np.testing.assert_allclose(k, [[f, 0, 959.5], [0, f, 539.5], [0, 0, 1]])


def test_rotation_is_proper():
    r = cam(eye_mm=(120.0, -250.0, 80.0), target_mm=(5.0, 3.0, -2.0)).R
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(r) == pytest.approx(1.0)


def test_target_projects_to_principal_point_and_up_is_up():
    c = cam(eye_mm=(120.0, -250.0, 80.0), target_mm=(5.0, 3.0, -2.0))
    uv = c.project(np.array([c.target_mm, np.add(c.target_mm, (0, 0, 10))]))
    np.testing.assert_allclose(uv[0], [c.K[0, 2], c.K[1, 2]], atol=1e-9)
    assert uv[1, 1] < uv[0, 1]  # world +z appears higher in the image (smaller v)


def test_camera_looking_along_y_has_world_x_to_the_right():
    c = cam(eye_mm=(0.0, -300.0, 0.0))
    u0, u1 = c.project(np.array([[0.0, 0, 0], [10.0, 0, 0]]))[:, 0]
    assert u1 > u0


@pytest.mark.parametrize(
    "kw",
    [
        {"eye_mm": (0.0, 0.0, 0.0)},  # eye == target
        {"eye_mm": (0.0, 0.0, 300.0)},  # looking straight down with up = +z
        {"up": (0.0, 0.0, 0.0)},
        {"pixel_size_um": 0.0},
        {"width": 5000},
        {"unknown_field": 1},
        {"eye_mm": (float("nan"), 0.0, 0.0)},
        {"eye_mm": (float("inf"), 0.0, 0.0)},
        {"focal_length_mm": float("inf")},
        {"name": "a/b"},
        {"name": "../x"},
        {"name": ".hidden"},
    ],
)
def test_invalid_camera_rejected(kw):
    with pytest.raises(ValidationError):
        cam(**kw)


def test_duplicate_names_rejected():
    with pytest.raises(ValidationError, match="duplicate"):
        Rig(cameras=[cam(), cam()])


def test_rig_round_trip(tmp_path):
    rig = ring(4, 300.0, 100.0)
    save_rig(rig, tmp_path / "rig.json")
    assert load_rig(tmp_path / "rig.json") == rig


def test_ring_geometry():
    rig = ring(6, 300.0, 100.0, width=640, height=480)
    assert [c.name for c in rig.cameras] == [f"cam{k:02d}" for k in range(6)]
    for c in rig.cameras:
        assert math.hypot(c.eye_mm[0], c.eye_mm[1]) == pytest.approx(300.0)
        assert c.eye_mm[2] == 100.0 and c.width == 640


def test_dome_geometry():
    rig = dome(12, 300.0, 20.0)
    for c in rig.cameras:
        assert np.linalg.norm(c.eye_mm) == pytest.approx(300.0)
        assert c.eye_mm[2] >= 300.0 * math.sin(math.radians(20.0)) - 1e-9


def test_dome_near_zenith_is_valid():
    assert len(dome(1, 300.0, 89.9).cameras) == 1


@pytest.mark.parametrize("bad", [{"n": 0}, {"min_elevation_deg": 90.0}])
def test_bad_preset_arguments(bad):
    args = {"n": 4, "radius_mm": 300.0, "min_elevation_deg": 20.0, **bad}
    with pytest.raises(ValueError):
        dome(**args)
