import numpy as np
import open3d as o3d
from trimesh.visual.color import uv_to_color

from heaicare_capture_sim.gui import _mesh_options, _to_trimesh
from heaicare_capture_sim.head import load_mesh, make_synthetic_head


def test_browser_mesh_keeps_obj_texture_orientation(tmp_path):
    # Standard OBJ: vt v = 1 is the image's top row. Texture: top red, bottom blue.
    img = np.zeros((64, 64, 3), np.uint8)
    img[:32, :, 0], img[32:, :, 2] = 255, 255
    o3d.io.write_image(str(tmp_path / "t.png"), o3d.geometry.Image(img))
    (tmp_path / "q.mtl").write_text("newmtl skin\nmap_Kd t.png\n")
    (tmp_path / "q.obj").write_text(
        "mtllib q.mtl\nv -50 0 -50\nv 50 0 -50\nv 50 0 50\nv -50 0 50\n"
        "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nusemtl skin\nf 1/1 2/2 3/3\nf 1/1 3/3 4/4\n"
    )
    mesh = _to_trimesh(load_mesh(tmp_path / "q.obj"))
    colours = uv_to_color(mesh.visual.uv, mesh.visual.material.baseColorTexture)
    top = mesh.vertices[:, 2] > 0
    assert (colours[top, 0] == 255).all() and (colours[~top, 2] == 255).all()


def test_browser_mesh_matches_geometry_and_alpha():
    head = make_synthetic_head()
    mesh = _to_trimesh(head)
    tris = np.asarray(head.mesh.triangles)
    np.testing.assert_array_equal(
        mesh.vertices[mesh.faces], np.asarray(head.mesh.vertices)[tris]
    )
    assert mesh.visual.material.alphaMode == "OPAQUE"
    rgba = np.zeros((4, 4, 4), np.uint8)
    card = _to_trimesh(head.__class__(head.mesh, rgba, "card"))
    assert card.visual.material.alphaMode == "BLEND"


def test_mesh_options_list_obj_files_in_subfolders(tmp_path):
    for name in ("b/b.obj", "a/model.obj", "c/c.ply", "loose.obj"):
        (tmp_path / name).parent.mkdir(exist_ok=True)
        (tmp_path / name).write_text("")
    objs = [str(tmp_path / "a/model.obj"), str(tmp_path / "b/b.obj")]
    assert _mesh_options(tmp_path, "synthetic") == ["synthetic", *objs]
    assert _mesh_options(tmp_path, objs[0]) == ["synthetic", *objs]
    assert _mesh_options(tmp_path, "/elsewhere/h.obj") == [
        "synthetic",
        *objs,
        "/elsewhere/h.obj",
    ]
    assert _mesh_options(tmp_path / "missing", "synthetic") == ["synthetic"]
