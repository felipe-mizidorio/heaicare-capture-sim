from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

from heaicare_capture_sim.head import (
    load_head,
    load_mesh,
    make_synthetic_head,
    write_ground_truth,
)


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


def _textured_obj(path, textures):
    mesh = o3d.geometry.TriangleMesh.create_sphere(1.0, 10, create_uv_map=True)
    mesh.textures = [o3d.geometry.Image(t) for t in textures]
    ids = np.arange(len(mesh.triangles), dtype=np.int32) % len(textures)
    mesh.triangle_material_ids = o3d.utility.IntVector(ids)
    assert o3d.io.write_triangle_mesh(str(path), mesh)
    return len(mesh.triangles)


def test_load_mesh_keeps_obj_texture(tmp_path):
    # Open3D reads an OBJ as [empty default material, real material]; the
    # loader must pick the one the triangles use.
    texture = (np.random.default_rng(0).random((16, 32, 3)) * 255).astype(np.uint8)
    _textured_obj(tmp_path / "h.obj", [texture])
    head = load_mesh(tmp_path / "h.obj")
    np.testing.assert_array_equal(head.texture, texture)
    assert head.extras == ()


def test_load_mesh_first_material_is_head_rest_are_extras(tmp_path):
    skin = np.full((16, 32, 3), 200, np.uint8)
    card = np.zeros((16, 32, 4), np.uint8)  # see-through texels: alpha kept
    card[..., 3] = 100
    eye = np.full((16, 32, 4), 255, np.uint8)  # fully opaque: alpha dropped
    n = _textured_obj(tmp_path / "h.obj", [skin, card, eye])
    head = load_mesh(tmp_path / "h.obj")
    np.testing.assert_array_equal(head.texture, skin)
    assert [e.texture.shape[2] for e in head.extras] == [4, 3]
    np.testing.assert_array_equal(head.extras[0].texture, card)
    counts = [len(h.mesh.triangles) for h in (head, *head.extras)]
    assert sum(counts) == n and min(counts) > 0
    assert all(
        len(h.mesh.triangle_uvs) == 3 * len(h.mesh.triangles) for h in head.extras
    )


def _quads_obj(path, parts):
    """One unit quad per (material, uv box) in ``parts``: [(texture, [boxes])]."""
    vertices, triangles, uvs, ids = [], [], [], []
    for k, (_, boxes) in enumerate(parts):
        for u0, u1 in boxes:
            n = len(vertices)
            x = float(len(triangles))
            vertices += [[x, 0, 0], [x + 1, 0, 0], [x + 1, 0, 1], [x, 0, 1]]
            triangles += [[n, n + 1, n + 2], [n, n + 2, n + 3]]
            uvs += [[u0, 0.2], [u1, 0.2], [u1, 0.8], [u0, 0.2], [u1, 0.8], [u0, 0.8]]
            ids += [k, k]
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles)
    )
    mesh.triangle_uvs = o3d.utility.Vector2dVector(uvs)
    mesh.textures = [o3d.geometry.Image(t) for t, _ in parts]
    mesh.triangle_material_ids = o3d.utility.IntVector(np.array(ids, np.int32))
    assert o3d.io.write_triangle_mesh(str(path), mesh)


def test_extras_drop_invisible_triangles(tmp_path):
    skin = np.full((8, 8, 3), 200, np.uint8)
    eye = np.full((8, 8, 4), 255, np.uint8)  # clear cornea texels on the left
    eye[:, :4, 3] = 0
    card = np.zeros((8, 8, 4), np.uint8)
    card[..., 3] = 100
    clear = np.zeros((8, 8, 4), np.uint8)
    _quads_obj(
        tmp_path / "h.obj",
        [
            (skin, [(0.1, 0.9)]),
            (eye, [(0.1, 0.4), (0.6, 0.9)]),  # cornea quad, eyeball quad
            (card, [(0.1, 0.9)]),
            (clear, [(0.1, 0.9)]),  # nothing visible: part dropped
        ],
    )
    head = load_mesh(tmp_path / "h.obj")
    eyeball, brow = head.extras
    assert len(eyeball.mesh.triangles) == 2 and eyeball.texture.shape[2] == 3
    assert len(brow.mesh.triangles) == 2 and brow.texture.shape[2] == 4
    assert len(head.mesh.triangles) == 2  # the head itself is never trimmed


def test_load_mesh_triangulates_obj_polygons(tmp_path):
    # Open3D silently drops non-triangle faces; a quad and a pentagon must survive.
    (tmp_path / "p.obj").write_text(
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv 2 0 0\nv 3 0 0\nv 3 1 0\n"
        "v 2.5 2 0\nv 2 1 0\nf 1 2 3 4\nf 5 6 7 8 9\n"
    )
    head = load_mesh(tmp_path / "p.obj")
    assert len(head.mesh.triangles) == 2 + 3
    assert head.mesh.get_surface_area() == pytest.approx(1.0 + 1.5)
    assert head.source == str(tmp_path / "p.obj")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["p.obj"]  # temp copy removed


def test_load_mesh_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_mesh(tmp_path / "missing.ply")
    (tmp_path / "empty.ply").write_text("not a mesh")
    with pytest.raises(ValueError, match="no triangles"):
        load_mesh(tmp_path / "empty.ply")


def test_write_ground_truth_errors(tmp_path):
    with pytest.raises(OSError, match="could not write ground truth"):
        write_ground_truth(make_synthetic_head(), tmp_path / "missing_dir" / "gt.ply")


def test_load_head_resolves_synthetic_names_and_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for folder, name in (("baby", "baby.obj"), ("scan", "model.obj")):
        (tmp_path / "meshes" / folder).mkdir(parents=True)
        write_ground_truth(make_synthetic_head(), tmp_path / "meshes" / folder / name)
    write_ground_truth(make_synthetic_head(), tmp_path / "other.ply")
    assert load_head("synthetic").source == "synthetic"
    assert load_head("baby").source == str(Path("meshes/baby/baby.obj"))
    assert load_head("scan").source == str(Path("meshes/scan/model.obj"))  # any name
    assert load_head("meshes/baby/baby.obj").source == "meshes/baby/baby.obj"
    assert load_head(tmp_path / "other.ply").source == str(tmp_path / "other.ply")
    with pytest.raises(FileNotFoundError, match="meshes/missing"):
        load_head("missing")
    write_ground_truth(make_synthetic_head(), tmp_path / "meshes" / "scan" / "b.obj")
    with pytest.raises(ValueError, match=r"meshes/scan.*one \.obj.*found 2"):
        load_head("scan")
