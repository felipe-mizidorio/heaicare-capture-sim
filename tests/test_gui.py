from concurrent.futures import Future

import numpy as np
import open3d as o3d
import viser
from trimesh.visual.color import uv_to_color

from heaicare_capture_sim.gui import _Gui, _mesh_options, _to_trimesh
from heaicare_capture_sim.head import load_mesh, make_synthetic_head
from heaicare_capture_sim.rig import ring
from heaicare_capture_sim.scenario import load_scenario


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


class _NoRender:
    def submit(self, job):
        future = Future()
        future.set_result(None)
        return future


def test_gizmos_move_camera_eye_and_room():
    server = viser.ViserServer(port=8123)  # viser moves to a free port if taken
    try:
        g = _Gui(server, _NoRender(), make_synthetic_head(), ring(4, 300.0, 100.0))
        # Handlers read the gizmo pose, as after a browser drag.
        g.cam_gizmo.position = (100.0, 200.0, 50.0)
        g._on_cam_gizmo(None)
        cam = g.rig.cameras[0]
        assert cam.eye_mm == (100.0, 200.0, 50.0) and cam.target_mm == (0, 0, 0)
        assert tuple(g.fields["eye_mm"].value) == cam.eye_mm
        g.cam_gizmo.position = (0.0, 0.0, 0.0)  # onto the target: invalid
        g._on_cam_gizmo(None)
        assert g.rig.cameras[0] == cam
        np.testing.assert_allclose(g.cam_gizmo.position, cam.eye_mm)  # snapped back
        g._select(2)
        np.testing.assert_allclose(g.cam_gizmo.position, g.rig.cameras[2].eye_mm)

        g.rig_gizmo.position = g._rig_centre() + (0.0, 0.0, 40.0)
        before = [c.eye_mm for c in g.rig.cameras]
        g._on_rig_gizmo(None)
        for b, c in zip(before, g.rig.cameras, strict=True):
            np.testing.assert_allclose(c.eye_mm, np.add(b, (0, 0, 40)), atol=1e-9)

        g._on_baby_gizmo(None)  # no scenario: nothing to move in
        assert g.scenario is None and not g.baby_gizmo.visible
        g._show_scenario(load_scenario("greybox"))
        assert g.baby_gizmo.visible
        room = np.asarray(g.scenario.parts[0][0].vertices).copy()
        g.baby_yaw.value = 90.0  # server-side set fires the field handler
        g.baby_gizmo.position = (500.0, 0.0, g.feet_z)
        g._on_baby_gizmo(None)
        assert g.scenario.spec.baby_offset_mm == (500.0, 0.0, 0.0)
        assert g.scenario.spec.baby_yaw_deg == 90.0
        assert tuple(g.baby_offset.value) == (500.0, 0.0, 0.0)
        # Head-frame room = inverse baby pose: shift by -500 x, then turn -90°.
        rz_minus90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # row-vector form
        np.testing.assert_allclose(
            np.asarray(g.scenario.parts[0][0].vertices),
            (room - (500.0, 0.0, 0.0)) @ rz_minus90,
            atol=1e-9,
        )
        np.testing.assert_allclose(g.baby_frame.position, (500.0, 0.0, 0.0))
        g._show_scenario(None)
        assert not g.baby_gizmo.visible
        np.testing.assert_allclose(g.baby_frame.position, (0.0, 0.0, 0.0))
    finally:
        server.stop()
