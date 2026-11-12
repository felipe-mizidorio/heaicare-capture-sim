"""Browser GUI (viser): edit a rig, preview any camera, capture all cameras with one button."""

from __future__ import annotations

import contextlib
import functools
import logging
import math
import threading
from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import trimesh
import viser
import viser.transforms as vtf
from PIL import Image
from pydantic import ValidationError

from heaicare_capture_sim.capture import CAPTURE_ROOT, capture, new_capture_dir
from heaicare_capture_sim.head import (
    MESH_DIR,
    SYNTHETIC,
    Head,
    load_head,
    load_mesh,
)
from heaicare_capture_sim.renderer import Renderer, RenderWorker
from heaicare_capture_sim.rig import (
    MAX_HEIGHT,
    MAX_WIDTH,
    Camera,
    Rig,
    dome,
    load_rig,
    move_rig,
    ring,
    save_rig,
)
from heaicare_capture_sim.scenario import (
    NONE,
    SCENARIO_DIR,
    Scenario,
    load_scenario,
    save_scenario,
)

_log = logging.getLogger(__name__)
PREVIEW_MAX_WIDTH = 480
_SKIN_RGB = (222, 171, 148)
_GIZMO_LIMITS = ((-10000.0, 10000.0),) * 3  # mm; viser's default is +-1000


def _to_trimesh(head: Head) -> trimesh.Trimesh:
    """One part of the head as a textured mesh for the browser's 3D view.

    Open3D stores UVs per triangle corner, so vertices are unwelded (one per
    corner). The texture is flipped exactly as ``Renderer.set_scene`` does.

    Parameters
    ----------
    head : Head
        A head or one of its extras (``extras`` are not followed).

    Returns
    -------
    trimesh.Trimesh
        Textured mesh, or a flat skin-coloured one if the part has no texture.
    """
    tris = np.asarray(head.mesh.triangles)
    vertices = np.asarray(head.mesh.vertices)[tris].reshape(-1, 3)
    faces = np.arange(len(vertices)).reshape(-1, 3)
    if head.texture is None or not head.mesh.has_triangle_uvs():
        return trimesh.Trimesh(vertices, faces, face_colors=_SKIN_RGB, process=False)
    see_through = head.texture.shape[2] == 4
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(np.ascontiguousarray(head.texture[::-1])),
        metallicFactor=0.0,
        roughnessFactor=1.0,
        alphaMode="BLEND" if see_through else "OPAQUE",
        doubleSided=see_through,  # hair cards are seen from both sides
    )
    visual = trimesh.visual.TextureVisuals(
        uv=np.asarray(head.mesh.triangle_uvs), material=material
    )
    return trimesh.Trimesh(vertices, faces, visual=visual, process=False)


def _mesh_options(mesh_dir: Path, current: str) -> list[str]:
    """Head choices: the synthetic head, each ``mesh_dir/<head>/*.obj``, and ``current``.

    Parameters
    ----------
    mesh_dir : Path
        Folder with one subfolder per ready-to-use head (world frame, mm).
    current : str
        Source of the head on screen (kept even if it lives elsewhere).

    Returns
    -------
    list of str
        "synthetic" first, then mesh paths sorted by name.
    """
    paths = sorted(str(p) for p in Path(mesh_dir).glob("*/*.obj"))
    return [
        SYNTHETIC,
        *paths,
        *([current] if current not in (SYNTHETIC, *paths) else []),
    ]


def _scenario_options(scenario_dir: Path, current: str) -> list[str]:
    """Scenario choices: none, each ``scenario_dir/<name>/scenario.json``, and ``current``.

    Parameters
    ----------
    scenario_dir : Path
        Folder with one subfolder per scenario.
    current : str
        Source of the scenario on screen (kept even if it lives elsewhere).

    Returns
    -------
    list of str
        "none" first, then scenario.json paths sorted by name.
    """
    paths = sorted(str(p) for p in Path(scenario_dir).glob("*/scenario.json"))
    return [NONE, *paths, *([current] if current not in (NONE, *paths) else [])]


def _first_head() -> Head:
    """The first head in ``MESH_DIR`` (by folder name), or the synthetic head.

    Returns
    -------
    Head
        The head the GUI starts on; a mesh that fails to load is skipped with a
        warning.
    """
    for name in _mesh_options(MESH_DIR, SYNTHETIC)[1:2]:
        try:
            return load_mesh(Path(name))
        except (ValueError, OSError):
            _log.warning("could not load %s; starting on the synthetic head", name)
    return load_head(SYNTHETIC)


def run(rig_path: Path | None, port: int) -> None:
    """Start the GUI server and block until interrupted.

    The head is picked in the GUI's Head panel; it starts on the first mesh in
    ``MESH_DIR``, or the synthetic head if there is none.

    Parameters
    ----------
    rig_path : Path or None
        Rig JSON to load; ``None`` starts from an 8-camera ring.
    port : int
        TCP port for the viser server.
    """
    rig = load_rig(rig_path) if rig_path else ring(8, 300.0, 100.0)
    MESH_DIR.mkdir(exist_ok=True)
    head = _first_head()
    worker = RenderWorker(lambda: Renderer(MAX_WIDTH, MAX_HEIGHT))
    server = None
    try:
        server = viser.ViserServer(port=port)
        if server.get_port() != port:  # viser silently moves to the next free port
            raise OSError(f"port {port} is already in use")
        _Gui(server, worker, head, rig)
        print(f"GUI at http://localhost:{port}")
        server.sleep_forever()
    except KeyboardInterrupt:
        pass
    finally:
        # Explicit shutdown: the renderer must be released on its own thread, so the
        # worker is closed even when the server failed to start.
        if server is not None:
            server.stop()
        worker.close()


def _notify(event, title: str, body: str) -> None:
    if event is not None and event.client is not None:
        event.client.add_notification(title=title, body=body)


def _locked(method: Callable) -> Callable:
    """Run a handler under the GUI lock (viser runs client callbacks on a thread pool)."""

    @functools.wraps(method)
    def wrapper(self: _Gui, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


class _Gui:
    def __init__(
        self, server: viser.ViserServer, worker: RenderWorker, head: Head, rig: Rig
    ) -> None:
        self.server, self.worker, self.head, self.rig = server, worker, head, rig
        self.selected, self.syncing, self.preview_gen = 0, False, 0
        self.frustums: list = []
        self.head_parts: list = []
        self.scenario: Scenario | None = None
        self.scenario_parts: list = []
        self.feet_z = 0.0
        self._lock = (
            threading.RLock()
        )  # reentrant: server-side .value sets re-enter handlers
        # The scene is in millimetres (head ~100 mm, cameras ~300 mm away), so the
        # client defaults (a few units from the origin, far plane 1000) show nothing.
        server.scene.set_up_direction("+z")
        cam = server.initial_camera
        cam.position, cam.look_at, cam.up = (
            (600.0, -600.0, 450.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        )
        cam.near, cam.far = 1.0, 10000.0
        # The browser shows the room fixed. Head, frustums and their gizmos are
        # drawn in the head frame under /baby, posed at the baby's place in the
        # room; outputs stay head-centred (the renderer gets the room moved instead).
        self.baby_frame = server.scene.add_frame("/baby", show_axes=False)
        self._draw_head()
        # Move-only gizmos; nested ones report head-frame positions.
        self.cam_gizmo = self._add_gizmo("/baby/camera_gizmo", 60.0, self._on_cam_gizmo)
        self.rig_gizmo = self._add_gizmo("/baby/rig_gizmo", 120.0, self._on_rig_gizmo)
        self.baby_gizmo = self._add_gizmo("/baby_gizmo", 200.0, self._on_baby_gizmo)
        self.baby_gizmo.visible = False
        gui = server.gui
        with gui.add_folder("Head"):
            self.mesh_dropdown = gui.add_dropdown(
                "Mesh",
                options=_mesh_options(MESH_DIR, head.source),
                initial_value=head.source,
            )
            self.mesh_dropdown.on_update(self._on_mesh)
            gui.add_button("Reload").on_click(self._rescan)
        with gui.add_folder("Scenario"):
            self.scenario_dropdown = gui.add_dropdown(
                "Scene",
                options=_scenario_options(SCENARIO_DIR, NONE),
                initial_value=NONE,
            )
            self.scenario_dropdown.on_update(self._on_scenario)
            gui.add_button("Reload scenario").on_click(self._rescan_scenario)
            self.baby_offset = gui.add_vector3(
                "Baby offset (mm)", initial_value=(0.0, 0.0, 0.0), step=1.0
            )
            self.baby_yaw = gui.add_number("Baby yaw (°)", initial_value=0.0, step=0.1)
            for handle in (self.baby_offset, self.baby_yaw):
                handle.on_update(self._on_baby_edit)
            gui.add_button("Save scenario").on_click(self._save_scenario)
        with gui.add_folder("Rig file"):
            self.rig_path = gui.add_text("Path", initial_value="rig.json")
            gui.add_button("Load rig").on_click(self._load)
            gui.add_button("Save rig").on_click(self._save)
        with gui.add_folder("Camera"):
            self.dropdown = gui.add_dropdown(
                "Selected", options=self._names(), initial_value=self._names()[0]
            )
            self.dropdown.on_update(self._on_select)
            self.fields = {
                "name": gui.add_text("Name", initial_value=""),
                "width": gui.add_number(
                    "Width (px)", initial_value=1920, min=1, max=MAX_WIDTH, step=1
                ),
                "height": gui.add_number(
                    "Height (px)", initial_value=1080, min=1, max=MAX_HEIGHT, step=1
                ),
                "focal_length_mm": gui.add_number(
                    "Focal (mm)", initial_value=8.0, min=0.1, step=0.1
                ),
                "pixel_size_um": gui.add_number(
                    "Pixel (µm)", initial_value=3.45, min=0.1, step=0.01
                ),
                "eye_mm": gui.add_vector3(
                    "Eye (mm)", initial_value=(0.0, 0.0, 0.0), step=1.0
                ),
                "target_mm": gui.add_vector3(
                    "Target (mm)", initial_value=(0.0, 0.0, 0.0), step=1.0
                ),
                "up": gui.add_vector3("Up", initial_value=(0.0, 0.0, 1.0), step=0.1),
            }
            for handle in self.fields.values():
                handle.on_update(self._on_edit)
            gui.add_button("Add").on_click(self._add)
            gui.add_button("Duplicate").on_click(self._duplicate)
            gui.add_button("Remove").on_click(self._remove)
        with gui.add_folder("Move all cameras"):
            self.rig_shift = gui.add_vector3(
                "Shift (mm)", initial_value=(0.0, 0.0, 0.0), step=1.0
            )
            self.rig_turn = gui.add_number("Turn (°)", initial_value=0.0, step=0.1)
            gui.add_button("Apply move").on_click(self._apply_move)
        with gui.add_folder("Ring preset"):
            self.ring_n = gui.add_number("Cameras", initial_value=8, min=1, step=1)
            self.ring_r = gui.add_number("Radius (mm)", initial_value=300.0, min=1.0)
            self.ring_h = gui.add_number("Height (mm)", initial_value=100.0)
            gui.add_button("Apply ring").on_click(self._apply_ring)
        with gui.add_folder("Dome preset"):
            self.dome_n = gui.add_number("Cameras", initial_value=12, min=1, step=1)
            self.dome_r = gui.add_number("Radius (mm)", initial_value=300.0, min=1.0)
            self.dome_e = gui.add_number(
                "Min elevation (°)", initial_value=20.0, min=0.0, max=89.9
            )
            gui.add_button("Apply dome").on_click(self._apply_dome)
        self.preview = gui.add_image(np.zeros((270, 480, 3), np.uint8), label="Preview")
        self.capture_button = gui.add_button("Capture")
        self.capture_button.on_click(self._capture)
        self._refresh()

    # state -> widgets
    def _names(self) -> list[str]:
        return [c.name for c in self.rig.cameras]

    def _refresh(self) -> None:
        cam = self.rig.cameras[self.selected]
        self.syncing = True
        try:
            self.dropdown.options = self._names()
            self.dropdown.value = cam.name
            for key, handle in self.fields.items():
                handle.value = getattr(cam, key)
            # Not while the user drags it: an echo of an older pose would make it jump.
            if not np.allclose(self.cam_gizmo.position, cam.eye_mm):
                self.cam_gizmo.position = cam.eye_mm
            centre = self._rig_centre()
            if not np.allclose(self.rig_gizmo.position, centre):
                self.rig_gizmo.position = centre
        finally:
            self.syncing = False
        self._draw_frustums()
        self._refresh_preview()

    def _add_gizmo(self, name: str, scale: float, handler: Callable):
        gizmo = self.server.scene.add_transform_controls(
            name,
            scale=scale,
            disable_rotations=True,  # turns are typed in the yaw / turn fields
            translation_limits=_GIZMO_LIMITS,
        )
        gizmo.on_update(handler)
        return gizmo

    def _rig_centre(self) -> np.ndarray:
        return np.mean([c.eye_mm for c in self.rig.cameras], axis=0)

    def _draw_head(self) -> None:
        for handle in self.head_parts:
            handle.remove()
        parts = (self.head, *self.head.extras)
        self.head_parts = [
            self.server.scene.add_mesh_trimesh(f"/baby/head/part{i}", _to_trimesh(part))
            for i, part in enumerate(parts)
        ]
        # The baby's gizmo sits at its lowest point (the feet, for a body).
        self.feet_z = min(float(np.asarray(p.mesh.vertices)[:, 2].min()) for p in parts)

    def _draw_scenario(self) -> None:
        for handle in self.scenario_parts:
            handle.remove()
        parts = self.scenario.parts if self.scenario else ()
        # Parts are moved around the head; the browser draws them in the room frame.
        pose = self.scenario.spec.baby_pose() if self.scenario else np.eye(4)
        self.scenario_parts = [
            self.server.scene.add_mesh_simple(
                f"/scenario/part{i}",
                vertices=np.asarray(mesh.vertices) @ pose[:3, :3].T + pose[:3, 3],
                faces=np.asarray(mesh.triangles),
                color=tuple(int(255 * c) for c in rgb),
                flat_shading=True,
            )
            for i, (mesh, rgb) in enumerate(parts)
        ]

    def _draw_frustums(self) -> None:
        for handle in self.frustums:
            handle.remove()
        self.frustums = []
        for i, cam in enumerate(self.rig.cameras):
            handle = self.server.scene.add_camera_frustum(
                f"/baby/cameras/{cam.name}",
                fov=2 * math.atan(cam.height / 2 / cam.focal_px),
                aspect=cam.width / cam.height,
                scale=20.0,
                wxyz=vtf.SO3.from_matrix(cam.R.T).wxyz,  # camera-to-world, OpenCV axes
                position=cam.eye_mm,
                color=(255, 80, 0) if i == self.selected else (90, 90, 90),
            )
            handle.on_click(lambda _event, i=i: self._select(i))
            self.frustums.append(handle)

    def _refresh_preview(self) -> None:
        self.preview_gen += 1
        gen, cam, head = self.preview_gen, self.rig.cameras[self.selected], self.head
        scenario = self.scenario

        def job(r: Renderer) -> np.ndarray | None:
            if gen != self.preview_gen:  # a newer edit superseded this one
                return None
            r.set_scene([head, *head.extras], scenario)
            return r.render(cam)

        self.worker.submit(job).add_done_callback(self._show_preview)

    def _show_preview(self, future: Future) -> None:
        try:
            image = future.result()
        except Exception:  # noqa: BLE001 - future callback must never raise; logged below
            _log.exception("preview failed")
            return
        if image is not None:
            step = max(1, math.ceil(image.shape[1] / PREVIEW_MAX_WIDTH))
            try:
                self.preview.image = image[::step, ::step]
            except Exception:  # noqa: BLE001 - the server may already be stopped
                _log.debug("preview update dropped", exc_info=True)

    # widget events -> state
    def _set_rig(self, cameras: list[Camera], selected: int, event=None) -> None:
        try:
            self.rig = Rig(cameras=cameras)
        except ValidationError as exc:
            _notify(event, "Invalid rig", str(exc))
        self.selected = min(selected, len(self.rig.cameras) - 1)
        self._refresh()

    @_locked
    def _select(self, index: int) -> None:
        self.selected = index
        self._refresh()

    @_locked
    def _on_select(self, _event) -> None:
        if not self.syncing:
            self._select(self._names().index(self.dropdown.value))

    @_locked
    def _on_edit(self, event) -> None:
        if self.syncing:
            return
        values = {k: h.value for k, h in self.fields.items()}
        values["width"], values["height"] = int(values["width"]), int(values["height"])
        try:
            cam = Camera.model_validate(values)
        except ValidationError as exc:
            _notify(event, "Invalid camera", str(exc))
            self._refresh()  # revert widgets to the last valid camera
            return
        cameras = list(self.rig.cameras)
        cameras[self.selected] = cam
        self._set_rig(cameras, self.selected, event)

    def _unique_name(self, base: str) -> str:
        names = set(self._names())
        return next(
            f"{base}_{i}"
            for i in range(1, len(names) + 2)
            if f"{base}_{i}" not in names
        )

    @_locked
    def _add(self, event) -> None:
        cam = Camera(name=self._unique_name("cam"), eye_mm=(300.0, 0.0, 100.0))
        self._set_rig([*self.rig.cameras, cam], len(self.rig.cameras), event)

    @_locked
    def _duplicate(self, event) -> None:
        src = self.rig.cameras[self.selected]
        cam = src.model_copy(update={"name": self._unique_name(src.name)})
        self._set_rig([*self.rig.cameras, cam], len(self.rig.cameras), event)

    @_locked
    def _remove(self, event) -> None:
        if len(self.rig.cameras) == 1:
            _notify(event, "Cannot remove", "A rig needs at least one camera.")
            return
        cameras = [c for i, c in enumerate(self.rig.cameras) if i != self.selected]
        self._set_rig(cameras, max(0, self.selected - 1), event)

    def _preset_fields(self) -> dict:
        """Camera settings that generated presets inherit from the selected camera."""
        cam = self.rig.cameras[self.selected]
        return {
            "target_mm": cam.target_mm,
            "width": cam.width,
            "height": cam.height,
            "focal_length_mm": cam.focal_length_mm,
            "pixel_size_um": cam.pixel_size_um,
        }

    @_locked
    def _apply_ring(self, event) -> None:
        try:
            rig = ring(
                int(self.ring_n.value),
                self.ring_r.value,
                self.ring_h.value,
                **self._preset_fields(),
            )
        except ValueError as exc:
            _notify(event, "Invalid ring", str(exc))
            return
        self._set_rig(rig.cameras, 0, event)

    @_locked
    def _apply_dome(self, event) -> None:
        try:
            rig = dome(
                int(self.dome_n.value),
                self.dome_r.value,
                self.dome_e.value,
                **self._preset_fields(),
            )
        except ValueError as exc:
            _notify(event, "Invalid dome", str(exc))
            return
        self._set_rig(rig.cameras, 0, event)

    def _sync_mesh_dropdown(self) -> None:
        self.syncing = True
        try:
            self.mesh_dropdown.options = _mesh_options(MESH_DIR, self.head.source)
            self.mesh_dropdown.value = self.head.source
        finally:
            self.syncing = False

    def _show_head(self, head: Head) -> None:
        _log.info("head: %s", head.source)
        self.head = head
        self._draw_head()
        self._sync_baby()  # the gizmo follows the new feet
        self._refresh_preview()

    @_locked
    def _rescan(self, event) -> None:
        """Reload the head on screen from disk, then rescan the mesh folder."""
        if self.head.source != SYNTHETIC:
            try:
                self._show_head(load_mesh(Path(self.head.source)))
            except (ValueError, OSError) as exc:
                _notify(event, "Reload failed; keeping the head on screen", str(exc))
        self._sync_mesh_dropdown()

    @_locked
    def _on_mesh(self, event) -> None:
        choice = self.mesh_dropdown.value
        if self.syncing or choice == self.head.source:
            return
        try:
            head = load_head(choice)
        except (ValueError, OSError) as exc:
            _notify(event, "Mesh load failed", str(exc))
            self._sync_mesh_dropdown()  # back to the head on screen
            return
        self._show_head(head)

    def _show_scenario(self, scenario: Scenario | None) -> None:
        _log.info("scenario: %s", scenario.source if scenario else NONE)
        self.scenario = scenario
        self._draw_scenario()
        self._sync_baby()
        self._refresh_preview()

    def _sync_baby(self, move_gizmo: bool = True) -> None:
        """Baby fields, /baby frame and baby gizmo <- ``self.scenario``."""
        spec = self.scenario.spec if self.scenario else None
        offset = spec.baby_offset_mm if spec else (0.0, 0.0, 0.0)
        yaw = spec.baby_yaw_deg if spec else 0.0
        half = math.radians(yaw) / 2
        self.syncing = True
        try:
            self.baby_offset.value, self.baby_yaw.value = offset, yaw
            self.baby_frame.position = offset
            self.baby_frame.wxyz = (math.cos(half), 0.0, 0.0, math.sin(half))
            self.baby_gizmo.visible = spec is not None  # no room, nothing to move in
            if move_gizmo:
                self.baby_gizmo.position = np.add(offset, (0.0, 0.0, self.feet_z))
        finally:
            self.syncing = False

    def _move_baby(self, offset_mm, yaw_deg: float, event, from_gizmo: bool) -> None:
        if self.syncing:
            return
        if self.scenario is None:
            _notify(event, "No scenario", "Pick a scenario to move the baby in.")
        else:
            try:
                self.scenario = self.scenario.with_baby(
                    tuple(float(v) for v in offset_mm), float(yaw_deg)
                )
            except ValueError as exc:
                _notify(event, "Invalid baby pose", str(exc))
        self._sync_baby(move_gizmo=not from_gizmo)
        self._refresh_preview()

    @_locked
    def _on_baby_edit(self, event) -> None:
        self._move_baby(self.baby_offset.value, self.baby_yaw.value, event, False)

    @_locked
    def _on_baby_gizmo(self, event) -> None:
        # Read the handle, not the event: thread-pool callbacks may run out of order.
        offset = np.subtract(self.baby_gizmo.position, (0.0, 0.0, self.feet_z))
        yaw = self.scenario.spec.baby_yaw_deg if self.scenario else 0.0
        self._move_baby(offset, yaw, event, True)

    @_locked
    def _save_scenario(self, event) -> None:
        if self.scenario is None:
            _notify(event, "Nothing to save", "No scenario is loaded.")
            return
        try:
            save_scenario(self.scenario)
        except OSError as exc:
            _notify(event, "Save failed", str(exc))
            return
        _notify(event, "Saved", self.scenario.source)

    @_locked
    def _on_cam_gizmo(self, event) -> None:
        if self.syncing:
            return
        cam = self.rig.cameras[self.selected]
        eye = tuple(float(v) for v in self.cam_gizmo.position)  # latest, see above
        try:
            cam = Camera.model_validate(cam.model_dump() | {"eye_mm": eye})
        except ValidationError as exc:
            _notify(event, "Invalid camera", str(exc))
            self._refresh()  # gizmo back to the last valid eye
            return
        cameras = list(self.rig.cameras)
        cameras[self.selected] = cam
        self._set_rig(cameras, self.selected, event)

    def _move_rig(self, shift_mm, turn_deg: float, event) -> None:
        try:
            rig = move_rig(self.rig, tuple(float(v) for v in shift_mm), float(turn_deg))
        except ValueError as exc:
            _notify(event, "Invalid move", str(exc))
            self._refresh()  # rig gizmo back to the cameras
            return
        self._set_rig(rig.cameras, self.selected, event)

    @_locked
    def _on_rig_gizmo(self, event) -> None:
        if not self.syncing:  # the gizmo sits at the eyes' mean; a drag shifts it
            self._move_rig(self.rig_gizmo.position - self._rig_centre(), 0.0, event)

    @_locked
    def _apply_move(self, event) -> None:
        self._move_rig(self.rig_shift.value, self.rig_turn.value, event)

    def _sync_scenario_dropdown(self) -> None:
        current = self.scenario.source if self.scenario else NONE
        self.syncing = True
        try:
            self.scenario_dropdown.options = _scenario_options(SCENARIO_DIR, current)
            self.scenario_dropdown.value = current
        finally:
            self.syncing = False

    @_locked
    def _on_scenario(self, event) -> None:
        choice = self.scenario_dropdown.value
        current = self.scenario.source if self.scenario else NONE
        if self.syncing or choice == current:
            return
        try:
            scenario = load_scenario(choice)
        except (ValueError, OSError) as exc:
            _notify(event, "Scenario load failed", str(exc))
            self._sync_scenario_dropdown()  # back to the scenario on screen
            return
        self._show_scenario(scenario)

    @_locked
    def _rescan_scenario(self, event) -> None:
        """Reload the scenario on screen from disk, then rescan the scenario folder."""
        if self.scenario is not None:
            try:
                self._show_scenario(load_scenario(Path(self.scenario.source)))
            except (ValueError, OSError) as exc:
                _notify(
                    event, "Reload failed; keeping the scenario on screen", str(exc)
                )
        self._sync_scenario_dropdown()

    @_locked
    def _load(self, event) -> None:
        try:
            rig = load_rig(Path(self.rig_path.value))
        except (ValueError, OSError) as exc:
            _notify(event, "Load failed", str(exc))
            return
        self._set_rig(rig.cameras, 0, event)

    @_locked
    def _save(self, event) -> None:
        try:
            save_rig(self.rig, Path(self.rig_path.value))
        except OSError as exc:
            _notify(event, "Save failed", str(exc))
            return
        _notify(event, "Saved", self.rig_path.value)

    @_locked
    def _capture(self, event) -> None:
        rig, head, scenario = self.rig, self.head, self.scenario
        try:
            out = new_capture_dir(CAPTURE_ROOT)
        except OSError as exc:
            _log.exception("capture dir failed")
            _notify(event, "Capture failed", str(exc))
            return
        self.capture_button.disabled = True

        def done(future: Future) -> None:
            try:
                self.capture_button.disabled = False
            except Exception:  # noqa: BLE001 - the server may already be stopped
                _log.debug("capture button re-enable dropped", exc_info=True)
            try:
                _notify(event, "Captured", str(future.result()))
            except Exception as exc:  # noqa: BLE001 - surfaced to the user and logged
                _log.exception("capture failed")
                with contextlib.suppress(OSError):
                    out.rmdir()  # no partial folder; no-op if it holds anything
                _notify(event, "Capture failed", str(exc))

        self.worker.submit(
            lambda r: capture(rig, head, out, r, scenario)
        ).add_done_callback(done)
