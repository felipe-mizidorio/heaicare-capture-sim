import numpy as np
import open3d as o3d
import pytest

from heaicare_capture_sim.cli import main
from heaicare_capture_sim.head import load_mesh
from heaicare_capture_sim.importer import MAKEHUMAN, import_mesh


def _sphere(radii, centre) -> o3d.geometry.TriangleMesh:
    s = o3d.geometry.TriangleMesh.create_sphere(1.0, 30)
    s.vertices = o3d.utility.Vector3dVector(
        np.asarray(s.vertices) * radii + np.asarray(centre)
    )
    return s


def _fake_makehuman(path):
    """A MakeHuman-like export in dm: y up, face +z, 7.2 dm tall.

    Skin: torso (below the neck cut), head (y 6.0..7.2, z -0.7..0.7), nose on
    +z shifted to +x so a mirrored conversion is caught. Then, as MakeHuman
    writes them, a second material: a see-through brow card made of one quad
    at y 6.9..7.0, z 0.65.
    """
    parts = [
        o3d.geometry.TriangleMesh.create_box(2.0, 5.0, 1.0).translate((-1.0, 0, -0.5)),
        _sphere((0.6, 0.6, 0.7), (0, 6.6, 0)),
        _sphere((0.1, 0.1, 0.1), (0.2, 6.6, 0.7)),  # nose tip at x 0.2, z 0.8
    ]
    mesh = parts[0]
    for p in parts[1:]:
        mesh += p
    n = len(mesh.triangles)
    mesh.triangle_uvs = o3d.utility.Vector2dVector(
        np.random.default_rng(0).random((3 * n, 2))
    )
    texture = (np.random.default_rng(1).random((32, 64, 3)) * 255).astype(np.uint8)
    mesh.textures = [o3d.geometry.Image(texture)]
    mesh.triangle_material_ids = o3d.utility.IntVector(np.zeros(n, np.int32))
    assert o3d.io.write_triangle_mesh(str(path), mesh)

    card = np.zeros((8, 8, 4), np.uint8)
    card[..., 3] = 100
    o3d.io.write_image(str(path.parent / "brow.png"), o3d.geometry.Image(card))
    lines = path.read_text().splitlines()
    nv = sum(line.startswith("v ") for line in lines)
    nt = sum(line.startswith("vt ") for line in lines)
    path.write_text(
        "\n".join(lines)
        + "\nv -0.2 6.9 0.65\nv 0.2 6.9 0.65\nv 0.2 7.0 0.65\nv -0.2 7.0 0.65\n"
        + "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nusemtl brow\nf "
        + " ".join(f"{nv + i}/{nt + i}" for i in range(1, 5))
        + "\n"
    )
    mtl = path.with_suffix(".mtl")
    mtl.write_text(mtl.read_text() + "\nnewmtl brow\nmap_Kd brow.png\n")
    return texture


def test_makehuman_preset_frame_scale_and_cut(tmp_path):
    texture = _fake_makehuman(tmp_path / "mh.obj")
    head = import_mesh(tmp_path / "mh.obj", tmp_path / "head.obj", **MAKEHUMAN)
    v = np.asarray(head.mesh.vertices)
    lo, hi = v.min(0), v.max(0)
    np.testing.assert_allclose((lo + hi) / 2, 0.0, atol=1e-9)  # centred
    assert hi[1] - lo[1] == pytest.approx(120.0)  # head length, nose included
    k = 120.0 / 1.5  # mm per dm-of-fake after scaling
    assert hi[2] - lo[2] == pytest.approx(1.2 * k, rel=1e-3)  # torso cut away
    nose, crown = v[v[:, 1].argmax()], v[v[:, 2].argmax()]
    assert abs(nose[2]) < 1  # face +y
    # MakeHuman +x -> world -x: a rotation, not a mirror
    assert nose[0] == pytest.approx(-0.2 * k, abs=0.5)
    # crown +z; the nose pushes the bbox centre 0.05 dm forward of it
    np.testing.assert_allclose(crown[:2], [0.0, -0.05 * k], atol=0.5)

    # The quad card is triangulated, kept as an extra (not skin), and moved
    # with the same transform as the skin.
    (card,) = head.extras
    assert len(card.mesh.triangles) == 2 and card.texture.shape[2] == 4
    c = np.asarray(card.mesh.vertices)
    np.testing.assert_allclose(c[:, 1], (0.65 - 0.05) * k, atol=1e-6)
    np.testing.assert_allclose(sorted({*c[:, 2].round(6)}), [24.0, 32.0], atol=1e-6)

    reloaded = load_mesh(tmp_path / "head.obj")
    np.testing.assert_allclose(np.asarray(reloaded.mesh.vertices), v, atol=1e-4)
    np.testing.assert_array_equal(reloaded.texture, texture)
    assert head.source == str(tmp_path / "head.obj")


def test_keep_body_renders_torso_but_ground_truth_is_unchanged(tmp_path):
    _fake_makehuman(tmp_path / "mh.obj")
    cut = import_mesh(tmp_path / "mh.obj", tmp_path / "cut.obj", **MAKEHUMAN)
    body = import_mesh(
        tmp_path / "mh.obj", tmp_path / "body.obj", **MAKEHUMAN, keep_body=True
    )
    np.testing.assert_allclose(
        np.asarray(body.mesh.vertices), np.asarray(cut.mesh.vertices), atol=1e-4
    )
    *_, torso = body.extras
    assert len(body.extras) == len(cut.extras) + 1 and torso.texture is not None
    k = 120.0 / 1.5  # mm per dm-of-fake after scaling
    # The torso is the fake's 5 dm box, from its feet up to the neck cut.
    lo, hi = torso.mesh.get_min_bound(), torso.mesh.get_max_bound()
    assert hi[2] - lo[2] == pytest.approx(5.0 * k, rel=1e-3)
    assert hi[2] <= cut.mesh.get_min_bound()[2] + 1e-3
    with pytest.raises(ValueError, match="neck_cut"):
        import_mesh(tmp_path / "mh.obj", tmp_path / "x.obj", keep_body=True)


def test_generic_scan_is_only_reoriented_and_centred(tmp_path):
    # A head scan in mm with up = +x and the face toward -z, off-centre. The
    # nose is shifted toward source -y, which is world +x (right = front x up).
    head = _sphere((50.0, 47.5, 60.0), (500.0, 20.0, 30.0))
    nose = _sphere((5.0, 5.0, 5.0), (500.0, 20.0 - 10.0, 30.0 - 60.0))
    assert o3d.io.write_triangle_mesh(str(tmp_path / "scan.obj"), head + nose)
    out = import_mesh(tmp_path / "scan.obj", tmp_path / "head.obj", up="+x", front="-z")
    v = np.asarray(out.mesh.vertices)
    lo, hi = v.min(0), v.max(0)
    np.testing.assert_allclose((lo + hi) / 2, 0.0, atol=1e-9)
    np.testing.assert_allclose(hi - lo, [95.0, 125.0, 100.0], atol=0.1)  # no rescale
    tip = v[v[:, 1].argmax()]
    assert tip[0] == pytest.approx(10.0, abs=0.1)  # face +y, nose on world +x
    assert len(out.mesh.triangles) == len((head + nose).triangles)  # no cut


def test_default_import_keeps_a_world_frame_mesh(tmp_path):
    mesh = _sphere((47.5, 60.0, 50.0), (0.0, 0.0, 0.0))
    assert o3d.io.write_triangle_mesh(str(tmp_path / "w.obj"), mesh)
    out = import_mesh(tmp_path / "w.obj", tmp_path / "head.obj")
    # Same point set (the OBJ reader reorders vertices; OBJ keeps ~6 digits).
    dist = o3d.geometry.PointCloud(out.mesh.vertices).compute_point_cloud_distance(
        o3d.geometry.PointCloud(mesh.vertices)
    )
    assert len(out.mesh.vertices) == len(mesh.vertices) and max(dist) < 1e-4


def test_import_keeps_vertex_colours_of_untextured_mesh(tmp_path):
    mesh = _sphere((47.5, 60.0, 50.0), (0.0, 0.0, 0.0))
    colours = np.random.default_rng(0).random((len(mesh.vertices), 3))
    mesh.vertex_colors = o3d.utility.Vector3dVector(colours)
    assert o3d.io.write_triangle_mesh(str(tmp_path / "c.obj"), mesh)
    out = import_mesh(tmp_path / "c.obj", tmp_path / "head.obj")
    assert out.texture is None and out.mesh.has_vertex_colors()
    # The OBJ reader reorders vertices, so match colours by position.
    tree = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(mesh.vertices))
    nearest = [tree.search_knn_vector_3d(v, 1)[1][0] for v in out.mesh.vertices]
    np.testing.assert_allclose(
        np.asarray(out.mesh.vertex_colors), colours[nearest], atol=1e-5
    )


def test_import_creates_the_output_folder(tmp_path):
    mesh = _sphere((47.5, 60.0, 50.0), (0.0, 0.0, 0.0))
    assert o3d.io.write_triangle_mesh(str(tmp_path / "w.obj"), mesh)
    out = tmp_path / "meshes" / "new_head" / "new_head.obj"
    import_mesh(tmp_path / "w.obj", out)
    assert out.is_file()


def test_import_rejects_bad_arguments(tmp_path):
    _fake_makehuman(tmp_path / "mh.obj")
    src, dst = tmp_path / "mh.obj", tmp_path / "head.obj"
    with pytest.raises(ValueError, match="units"):
        import_mesh(src, dst, units="furlong")
    for up, front in (("y", "+z"), ("+y", "up"), ("+y", "-y"), ("+z", "+z")):
        with pytest.raises(ValueError, match="up|front"):
            import_mesh(src, dst, up=up, front=front)
    for cut in (0.0, 1.0, float("nan")):
        with pytest.raises(ValueError, match="neck_cut"):
            import_mesh(src, dst, neck_cut=cut)
    for length in (0.0, -1.0, float("inf")):
        with pytest.raises(ValueError, match="head_length_mm"):
            import_mesh(src, dst, head_length_mm=length)
    with pytest.raises(ValueError, match=r"\.obj"):
        import_mesh(src, tmp_path / "head.ply")
    with pytest.raises(FileNotFoundError):
        import_mesh(tmp_path / "missing.obj", dst)


def test_cli_import_mesh_preset_and_flags(tmp_path, capsys):
    _fake_makehuman(tmp_path / "mh.obj")
    src, out = str(tmp_path / "mh.obj"), tmp_path / "head.obj"
    preset = ["import-mesh", src, "-o", str(out), "--preset", "makehuman"]
    assert main([*preset, "--head-length-mm", "110"]) == 0  # flag overrides preset
    assert np.ptp(np.asarray(load_mesh(out).mesh.vertices)[:, 1]) == pytest.approx(
        110.0
    )
    assert main(preset) == 0
    assert np.ptp(np.asarray(load_mesh(out).mesh.vertices)[:, 1]) == pytest.approx(
        120.0
    )
    generic = ["import-mesh", src, "-o", str(out), "--units", "dm", "--up", "+y"]
    assert main([*generic, "--front", "+z"]) == 0  # no cut, no rescale: whole body
    assert np.ptp(np.asarray(load_mesh(out).mesh.vertices)[:, 2]) == pytest.approx(
        720.0
    )
    assert main([*preset, "--neck-cut", "2"]) == 2
    assert "neck_cut" in capsys.readouterr().err
