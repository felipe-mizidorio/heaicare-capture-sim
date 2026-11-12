"""One capture: render every camera of a rig and write the output folder atomically."""

from __future__ import annotations

import itertools
import json
import logging
import shutil
import tempfile
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import open3d as o3d

from heaicare_capture_sim.head import Head, write_ground_truth
from heaicare_capture_sim.rig import Rig, save_rig
from heaicare_capture_sim.scenario import Scenario

_log = logging.getLogger(__name__)


def require_empty_dir(path: Path) -> None:
    """Raise FileExistsError if ``path`` exists and is not an empty directory."""
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"output folder is not empty: {path}")


def new_capture_dir(root: Path) -> Path:
    """Create and return a new empty timestamped folder under ``root``."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for i in itertools.count():
        path = Path(root) / (stamp if i == 0 else f"{stamp}-{i}")
        try:
            path.mkdir(parents=True)
            return path
        except FileExistsError:
            continue
    raise AssertionError("unreachable")


def capture(
    rig: Rig, head: Head, out_dir: Path, renderer, scenario: Scenario | None = None
) -> Path:
    """Render every camera and write images, cameras.json, rig.json, ground_truth.ply.

    Parameters
    ----------
    rig : Rig
        Cameras to trigger together.
    head : Head
        Scene geometry; its extras are rendered, only ``head.mesh`` is ground truth.
    out_dir : Path
        Missing or empty folder; parents are created.
    renderer
        Object with ``background_rgb``, ``set_scene(meshes, scenario)`` and
        ``render(cam) -> (H, W, 3) uint8``.
    scenario : Scenario or None, optional
        Room rendered around the head, never in the ground truth (default None).

    Returns
    -------
    Path
        ``out_dir``, fully written. Nothing is left behind on failure.
    """
    out_dir = Path(out_dir)
    require_empty_dir(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.", dir=out_dir.parent))
    try:
        (tmp / "images").mkdir()
        renderer.set_scene([head, *head.extras], scenario)
        cameras = []
        for cam in rig.cameras:
            image = renderer.render(cam)
            if not (image != renderer.background_rgb).any():
                _log.warning("camera %s sees no geometry", cam.name)
            rel = f"images/{cam.name}.png"
            image_ok = o3d.io.write_image(
                str(tmp / rel), o3d.geometry.Image(np.ascontiguousarray(image))
            )
            if not image_ok:
                raise OSError(f"could not write image: {out_dir / rel}")
            cameras.append(
                {
                    "name": cam.name,
                    "image": rel,
                    "width": cam.width,
                    "height": cam.height,
                    "K": cam.K.tolist(),
                    "R": cam.R.tolist(),
                    "t_mm": cam.t_mm.tolist(),
                }
            )
        meta = {
            "created": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "version": version("heaicare-capture-sim"),
            "mesh_source": head.source,
            "scenario_source": scenario.source if scenario else None,
            "scenario": scenario.spec.model_dump(mode="json") if scenario else None,
            "cameras": cameras,
        }
        (tmp / "cameras.json").write_text(json.dumps(meta, indent=2) + "\n")
        save_rig(rig, tmp / "rig.json")
        write_ground_truth(head, tmp / "ground_truth.ply")
        if out_dir.exists():
            out_dir.rmdir()
        tmp.rename(out_dir)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return out_dir
