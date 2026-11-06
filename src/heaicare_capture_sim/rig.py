"""Camera rig schema, OpenCV camera matrices and layout presets."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Vec3 = tuple[float, float, float]
MAX_WIDTH, MAX_HEIGHT = 4096, 3072
_PARALLEL_TOL = 1e-6
_GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))


def _unit(v: Any) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        raise ValueError("zero-length vector")
    return v / norm


class Camera(BaseModel):
    """Ideal pinhole camera from datasheet values and a look-at pose.

    Lengths are millimetres in the world frame (z up). Square pixels, principal
    point at the image centre, OpenCV axes (x right, y down, z forward).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    width: int = Field(default=1920, gt=0, le=MAX_WIDTH)
    height: int = Field(default=1080, gt=0, le=MAX_HEIGHT)
    focal_length_mm: float = Field(default=8.0, gt=0)
    pixel_size_um: float = Field(default=3.45, gt=0)
    eye_mm: Vec3
    target_mm: Vec3 = (0.0, 0.0, 0.0)
    up: Vec3 = (0.0, 0.0, 1.0)

    @model_validator(mode="after")
    def _check_pose(self) -> Camera:
        forward = np.subtract(self.target_mm, self.eye_mm)
        if np.linalg.norm(forward) < 1e-9:
            raise ValueError(f"camera {self.name!r}: eye_mm equals target_mm")
        if np.linalg.norm(np.cross(_unit(forward), _unit(self.up))) < _PARALLEL_TOL:
            raise ValueError(
                f"camera {self.name!r}: up is parallel to the viewing direction"
            )
        return self

    @property
    def focal_px(self) -> float:
        """Focal length in pixels."""
        return self.focal_length_mm * 1000.0 / self.pixel_size_um

    @property
    def K(self) -> np.ndarray:
        """OpenCV intrinsic matrix (3, 3)."""
        f = self.focal_px
        return np.array(
            [
                [f, 0.0, (self.width - 1) / 2],
                [0.0, f, (self.height - 1) / 2],
                [0.0, 0.0, 1.0],
            ]
        )

    @property
    def R(self) -> np.ndarray:
        """World-to-camera rotation (3, 3), OpenCV axes."""
        z = _unit(np.subtract(self.target_mm, self.eye_mm))
        x = _unit(np.cross(z, self.up))
        return np.stack([x, np.cross(z, x), z])

    @property
    def t_mm(self) -> np.ndarray:
        """World-to-camera translation (3,) in mm."""
        return -self.R @ np.asarray(self.eye_mm, dtype=float)

    def project(self, points_mm: np.ndarray) -> np.ndarray:
        """Project world points.

        Parameters
        ----------
        points_mm : np.ndarray
            World points, shape (N, 3), in mm.

        Returns
        -------
        np.ndarray
            Pixel coordinates (u, v), shape (N, 2), pixel-centre convention.
        """
        cam = np.asarray(points_mm, dtype=float) @ self.R.T + self.t_mm
        uvw = cam @ self.K.T
        return uvw[:, :2] / uvw[:, 2:3]


class Rig(BaseModel):
    """An explicit list of cameras triggered together."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    cameras: list[Camera] = Field(min_length=1)

    @field_validator("cameras")
    @classmethod
    def _unique_names(cls, cameras: list[Camera]) -> list[Camera]:
        names = [c.name for c in cameras]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate camera names: {dupes}")
        return cameras


def load_rig(path: Path) -> Rig:
    """Read and validate a rig JSON file."""
    return Rig.model_validate_json(Path(path).read_text())


def save_rig(rig: Rig, path: Path) -> None:
    """Write a rig as indented JSON."""
    Path(path).write_text(rig.model_dump_json(indent=2) + "\n")


def ring(
    n: int,
    radius_mm: float,
    height_mm: float,
    target_mm: Vec3 = (0.0, 0.0, 0.0),
    **camera_fields: Any,
) -> Rig:
    """Place n cameras evenly on a horizontal circle around the z axis through target.

    Parameters
    ----------
    n : int
        Number of cameras (>= 1).
    radius_mm : float
        Horizontal distance from the target's z axis.
    height_mm : float
        World z of the ring plane.
    target_mm : Vec3
        Point every camera looks at.
    **camera_fields
        Shared ``Camera`` fields (width, height, focal_length_mm, pixel_size_um).
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    cams = []
    for k in range(n):
        a = 2.0 * math.pi * k / n
        eye = (
            target_mm[0] + radius_mm * math.cos(a),
            target_mm[1] + radius_mm * math.sin(a),
            height_mm,
        )
        cams.append(
            Camera(name=f"cam{k:02d}", eye_mm=eye, target_mm=target_mm, **camera_fields)
        )
    return Rig(cameras=cams)


def dome(
    n: int,
    radius_mm: float,
    min_elevation_deg: float,
    target_mm: Vec3 = (0.0, 0.0, 0.0),
    **camera_fields: Any,
) -> Rig:
    """Spread n cameras evenly (Fibonacci) over the hemisphere cap above target.

    Parameters
    ----------
    n : int
        Number of cameras (>= 1).
    radius_mm : float
        Distance from target to every camera.
    min_elevation_deg : float
        Lowest elevation above the target's horizontal plane, in [0, 90).
    target_mm : Vec3
        Point every camera looks at.
    **camera_fields
        Shared ``Camera`` fields.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if not 0.0 <= min_elevation_deg < 90.0:
        raise ValueError("min_elevation_deg must be in [0, 90)")
    s_min = math.sin(math.radians(min_elevation_deg))
    cams = []
    for k in range(n):
        s = s_min + (1.0 - s_min) * (k + 0.5) / n  # equal-area bands in sin(elevation)
        c, a = math.sqrt(1.0 - s * s), k * _GOLDEN_ANGLE
        offset = (
            radius_mm * c * math.cos(a),
            radius_mm * c * math.sin(a),
            radius_mm * s,
        )
        eye = tuple(float(t + o) for t, o in zip(target_mm, offset, strict=True))
        cams.append(
            Camera(name=f"cam{k:02d}", eye_mm=eye, target_mm=target_mm, **camera_fields)
        )
    return Rig(cameras=cams)
