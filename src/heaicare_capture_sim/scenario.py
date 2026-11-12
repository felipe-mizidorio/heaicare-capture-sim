"""Scenarios: room geometry rendered around the head, never part of the ground truth."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
from pydantic import BaseModel, ConfigDict, field_validator

from heaicare_capture_sim.importer import AXES, UNITS_MM, to_world
from heaicare_capture_sim.rig import Vec3

SCENARIO_DIR = Path(
    "scenarios"
)  # one folder per scenario: scenarios/<name>/scenario.json
NONE = "none"


class ScenarioSpec(BaseModel):
    """``scenario.json``: a model file and where the head sits in it.

    ``up`` and ``front`` are the model's axes that become world +z (crown) and
    +y (face); ``head`` is the model point, in model units, that becomes the
    world origin (head centre). ``baby_offset_mm`` and ``baby_yaw_deg`` then
    move the baby (head and rig together) inside the room: a turn about the
    vertical through ``head`` and a shift, in room axes (both 0 by default; set
    by the GUI). Outputs stay head-centred, so the room is moved the other way.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    file: str
    units: str
    up: str
    front: str
    head: tuple[float, float, float]
    baby_offset_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    baby_yaw_deg: float = 0.0

    @field_validator("file")
    @classmethod
    def _plain_name(cls, v: str) -> str:
        if not v or Path(v).name != v:
            raise ValueError(
                f"file must be a file name next to scenario.json, got {v!r}"
            )
        return v

    @field_validator("units")
    @classmethod
    def _known_units(cls, v: str) -> str:
        if v not in UNITS_MM:
            raise ValueError(f"units must be one of {sorted(UNITS_MM)}, got {v!r}")
        return v

    @field_validator("up", "front")
    @classmethod
    def _known_axis(cls, v: str) -> str:
        if v not in AXES:
            raise ValueError(f"axis must be one of {sorted(AXES)}, got {v!r}")
        return v

    def baby_pose(self) -> np.ndarray:
        """4x4 head frame -> room frame: ``baby_yaw_deg`` about z, then the offset.

        The room frame is the world frame of an unmoved baby (``head`` at the
        origin), so the identity means the baby sits at ``head``.

        Returns
        -------
        np.ndarray
            Rigid transform.
        """
        yaw = np.radians(self.baby_yaw_deg)
        t = np.eye(4)
        t[:2, :2] = ((np.cos(yaw), -np.sin(yaw)), (np.sin(yaw), np.cos(yaw)))
        t[:3, 3] = self.baby_offset_mm
        return t

    def to_world(self) -> np.ndarray:
        """4x4 transform from model coordinates to the world frame (mm).

        Returns
        -------
        np.ndarray
            Scaled proper rotation plus translation; ``head`` maps to the origin,
            then the room is moved by the inverse of ``baby_pose()``.

        Raises
        ------
        ValueError
            If ``up`` and ``front`` are not perpendicular.
        """
        rs = to_world(self.up, self.front) * UNITS_MM[self.units]
        t = np.eye(4)
        t[:3, :3], t[:3, 3] = rs, -rs @ np.asarray(self.head)
        return np.linalg.inv(self.baby_pose()) @ t


@dataclass(frozen=True)
class Scenario:
    """Scenario geometry in the world frame, mm.

    Attributes
    ----------
    parts : tuple of (o3d.geometry.TriangleMesh, tuple of float)
        One mesh per model part with its material's base colour (RGB, 0..1).
    source : str
        Path of the ``scenario.json``.
    spec : ScenarioSpec
        The spec the parts were placed with.
    """

    parts: tuple[tuple[o3d.geometry.TriangleMesh, tuple[float, float, float]], ...]
    source: str
    spec: ScenarioSpec

    def with_baby(self, offset_mm: Vec3, yaw_deg: float) -> Scenario:
        """The same scenario with the baby at a new pose in the room.

        Parameters
        ----------
        offset_mm : Vec3
            New ``ScenarioSpec.baby_offset_mm``.
        yaw_deg : float
            New ``ScenarioSpec.baby_yaw_deg``.

        Returns
        -------
        Scenario
            Parts moved around the head; ``self`` is unchanged.

        Raises
        ------
        ValueError
            If a value is NaN or infinite.
        """
        spec = ScenarioSpec.model_validate(
            self.spec.model_dump()
            | {"baby_offset_mm": offset_mm, "baby_yaw_deg": yaw_deg}
        )
        move = np.linalg.inv(spec.baby_pose()) @ self.spec.baby_pose()
        parts = tuple(
            (o3d.geometry.TriangleMesh(mesh).transform(move), rgb)
            for mesh, rgb in self.parts
        )
        return Scenario(parts, self.source, spec)


def save_scenario(scenario: Scenario) -> None:
    """Write the scenario's spec, baby pose included, back to its ``scenario.json``.

    Parameters
    ----------
    scenario : Scenario
        Scenario whose ``source`` is the file to overwrite. Default baby pose
        fields are left out, so an unmoved baby keeps the original file.
    """
    path = Path(scenario.source)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        scenario.spec.model_dump_json(indent=2, exclude_defaults=True) + "\n"
    )
    tmp.replace(path)  # atomic: a crash never leaves a half-written scenario.json


def load_scenario(name: str | Path) -> Scenario | None:
    """No scenario, a folder in ``SCENARIO_DIR`` by name, or a ``scenario.json`` path.

    The model is read with Open3D (assimp: FBX, OBJ, glTF, PLY...). Only base
    colours are used; model textures are ignored.

    Parameters
    ----------
    name : str or Path
        "none"; a bare name such as "greybox" for
        ``SCENARIO_DIR / "greybox" / "scenario.json"``; or a path to a
        ``scenario.json``.

    Returns
    -------
    Scenario or None
        None for "none".

    Raises
    ------
    FileNotFoundError
        If the scenario or its model file does not exist.
    ValueError
        If ``scenario.json`` is invalid or the model has no triangles.
    """
    if str(name) == NONE:
        return None
    path = Path(name)
    if path.parent == Path(".") and not path.suffix:
        path = SCENARIO_DIR / path / "scenario.json"
    if not path.is_file():
        raise FileNotFoundError(f"scenario not found: {path}")
    spec = ScenarioSpec.model_validate_json(path.read_text())
    transform = spec.to_world()
    model_path = path.parent / spec.file
    if not model_path.is_file():
        raise FileNotFoundError(f"scenario model not found: {model_path}")
    model = o3d.io.read_triangle_model(str(model_path))
    parts = []
    for info in model.meshes:
        mesh = info.mesh.transform(transform)
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        rgb = tuple(float(c) for c in model.materials[info.material_idx].base_color[:3])
        parts.append((mesh, rgb))
    if not any(len(mesh.triangles) for mesh, _ in parts):
        raise ValueError(f"{model_path}: no triangles")
    return Scenario(tuple(parts), str(path), spec)
