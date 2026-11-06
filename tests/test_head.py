import numpy as np
import open3d as o3d
import pytest

from heaicare_capture_sim.head import load_mesh, make_synthetic_head, write_ground_truth


def test_synthetic_head_size_and_orientation():
    box = make_synthetic_head().mesh.get_axis_aligned_bounding_box()
    np.testing.assert_allclose(
        box.get_extent(), [95.0, 120.0, 100.0], atol=0.5
    )  # W(x), L(y), H(z)
    np.testing.assert_allclose(box.get_center(), [0.0, 0.0, 0.0], atol=0.5)


def test_synthetic_head_texture_is_deterministic_and_varied():
    a, b = make_synthetic_head(seed=0), make_synthetic_head(seed=0)
    assert a.texture.dtype == np.uint8 and a.texture.ndim == 3
    np.testing.assert_array_equal(a.texture, b.texture)
    assert a.texture.std(axis=(0, 1)).max() > 10
    assert a.mesh.has_triangle_uvs() and a.source == "synthetic"


def test_ground_truth_is_watertight_geometry(tmp_path):
    write_ground_truth(make_synthetic_head(), tmp_path / "gt.ply")
    gt = o3d.io.read_triangle_mesh(str(tmp_path / "gt.ply"))
    assert gt.is_watertight()
    assert not gt.has_vertex_colors() and not gt.has_triangle_uvs()


def test_load_mesh_round_trip(tmp_path):
    write_ground_truth(make_synthetic_head(), tmp_path / "gt.ply")
    head = load_mesh(tmp_path / "gt.ply")
    assert len(head.mesh.triangles) > 0 and head.texture is None
    assert head.source == str(tmp_path / "gt.ply")


def test_load_mesh_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_mesh(tmp_path / "missing.ply")
    (tmp_path / "empty.ply").write_text("not a mesh")
    with pytest.raises(ValueError, match="no triangles"):
        load_mesh(tmp_path / "empty.ply")
