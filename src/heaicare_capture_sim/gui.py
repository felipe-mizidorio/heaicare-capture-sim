"""Browser GUI (viser): edit a rig, preview any camera, capture all cameras with one button."""

from __future__ import annotations

import functools
import logging
import math
import threading
from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import viser
import viser.transforms as vtf
from pydantic import ValidationError

from heaicare_capture_sim.capture import capture, new_capture_dir
from heaicare_capture_sim.head import Head, load_mesh, make_synthetic_head
from heaicare_capture_sim.renderer import Renderer, RenderWorker
from heaicare_capture_sim.rig import (
    MAX_HEIGHT,
    MAX_WIDTH,
    Camera,
    Rig,
    dome,
    load_rig,
    ring,
    save_rig,
)

_log = logging.getLogger(__name__)
CAPTURE_ROOT = Path("captures")
PREVIEW_MAX_WIDTH = 480
_SKIN_RGB = (222, 171, 148)


def run(rig_path: Path | None, mesh_path: Path | None, port: int) -> None:
    """Start the GUI server and block until interrupted.

    Parameters
    ----------
    rig_path : Path or None
        Rig JSON to load; ``None`` starts from an 8-camera ring.
    mesh_path : Path or None
        Head mesh to load; ``None`` uses the synthetic head.
    port : int
        TCP port for the viser server.
    """
    head = load_mesh(mesh_path) if mesh_path else make_synthetic_head()
    rig = load_rig(rig_path) if rig_path else ring(8, 300.0, 100.0)
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
        server.scene.add_mesh_simple(
            "/head",
            vertices=np.asarray(head.mesh.vertices, dtype=np.float32),
            faces=np.asarray(head.mesh.triangles, dtype=np.uint32),
            color=_SKIN_RGB,
        )
        gui = server.gui
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
        finally:
            self.syncing = False
        self._draw_frustums()
        self._refresh_preview()

    def _draw_frustums(self) -> None:
        for handle in self.frustums:
            handle.remove()
        self.frustums = []
        for i, cam in enumerate(self.rig.cameras):
            handle = self.server.scene.add_camera_frustum(
                f"/cameras/{cam.name}",
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

        def job(r: Renderer) -> np.ndarray | None:
            if gen != self.preview_gen:  # a newer edit superseded this one
                return None
            r.set_scene([head])
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
        rig, head = self.rig, self.head
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
                _notify(event, "Capture failed", str(exc))

        self.worker.submit(lambda r: capture(rig, head, out, r)).add_done_callback(done)
