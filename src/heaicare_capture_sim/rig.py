"""Camera rig schema, OpenCV camera matrices and layout presets."""

from __future__ import annotations

import functools
import math
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Vec3 = tuple[float, float, float]
MAX_WIDTH, MAX_HEIGHT = 4096, 3072
_PARALLEL_TOL = 1e-6
_GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))
_UNDISTORT_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-12)
_ROUND_TRIP_TOL_PX = 1e-3
_CHECK_STEP_PX = 8  # lens validation samples every 8th pixel (and the last row/column)


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
    ``dist`` is OpenCV's (k1, k2, p1, p2, k3); ``blur_px``, ``gain`` and
    ``noise_std`` are applied to the rendered image (see ``capture.apply_sensor``).
    All four default to an ideal camera.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    width: int = Field(default=1920, gt=0, le=MAX_WIDTH)
    height: int = Field(default=1080, gt=0, le=MAX_HEIGHT)
    focal_length_mm: float = Field(default=8.0, gt=0)
    pixel_size_um: float = Field(default=3.45, gt=0)
    eye_mm: Vec3
    target_mm: Vec3 = (0.0, 0.0, 0.0)
    up: Vec3 = (0.0, 0.0, 1.0)
    dist: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)
    blur_px: float = Field(default=0.0, ge=0)
    gain: float = Field(default=1.0, gt=0)
    noise_std: float = Field(default=0.0, ge=0)

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

    @model_validator(mode="after")
    def _check_lens(self) -> Camera:
        if not any(self.dist):
            return self
        grid = _pixel_grid(self.width, self.height, _CHECK_STEP_PX)
        ideal = ideal_pixels(
            self.width, self.height, self.focal_px, self.dist, _CHECK_STEP_PX
        )
        # Past the fold radius pixels have no preimage (round trip fails) or
        # neighbours see the scene out of order.
        folded = (np.diff(ideal[..., 0], axis=1) <= 0).any() or (
            np.diff(ideal[..., 1], axis=0) <= 0
        ).any()
        error = np.abs(_distort(self.K, self.dist, ideal) - grid).max()
        if folded or error > _ROUND_TRIP_TOL_PX:
            raise ValueError(
                f"camera {self.name!r}: distortion {self.dist} is not invertible "
                "inside the image (too strong)"
            )
        return self

    @property
    def focal_px(self) -> float:
        """Focal length in pixels."""
        return self.focal_length_mm * 1000.0 / self.pixel_size_um

    @property
    def K(self) -> np.ndarray:
        """OpenCV intrinsic matrix (3, 3)."""
        return _k(self.width, self.height, self.focal_px)

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
            Pixel coordinates (u, v), shape (N, 2), pixel-centre convention,
            lens distortion included.
        """
        cam = np.asarray(points_mm, dtype=float) @ self.R.T + self.t_mm
        uv, _ = cv2.projectPoints(cam, np.zeros(3), np.zeros(3), self.K, self.dist)
        return uv.reshape(-1, 2)

    def ideal_pixels(self) -> np.ndarray:
        """Undistorted pinhole position of every pixel centre.

        Returns
        -------
        np.ndarray
            Shape (height, width, 2): (u, v) in the same ``K`` without distortion.
            Read-only and cached per lens.
        """
        return ideal_pixels(self.width, self.height, self.focal_px, self.dist, 1)


def _k(width: int, height: int, focal_px: float) -> np.ndarray:
    return np.array(
        [[focal_px, 0.0, (width - 1) / 2], [0.0, focal_px, (height - 1) / 2], [0, 0, 1]]
    )


@functools.lru_cache(maxsize=4)  # a 1920x1080 map is 33 MB
def ideal_pixels(
    width: int, height: int, focal_px: float, dist: tuple[float, ...], step: int
) -> np.ndarray:
    """Undistort pixel centres every ``step`` px (last row/column always included).

    Parameters
    ----------
    width, height : int
        Image size, px.
    focal_px : float
        Focal length, px.
    dist : tuple of float
        OpenCV (k1, k2, p1, p2, k3).
    step : int
        Sampling step, px.

    Returns
    -------
    np.ndarray
        Shape (rows, cols, 2), read-only.
    """
    grid = _pixel_grid(width, height, step)
    k = _k(width, height, focal_px)
    ideal = cv2.undistortPoints(
        grid.reshape(-1, 1, 2), k, np.array(dist), P=k, criteria=_UNDISTORT_CRITERIA
    ).reshape(grid.shape)
    ideal.flags.writeable = False
    return ideal


def _pixel_grid(width: int, height: int, step: int) -> np.ndarray:
    """Pixel centres every ``step`` px, last row/column included; shape (rows, cols, 2)."""
    us = np.unique(np.r_[0:width:step, width - 1])
    vs = np.unique(np.r_[0:height:step, height - 1])
    return np.stack(np.meshgrid(us.astype(float), vs.astype(float)), axis=-1)


def _distort(k: np.ndarray, dist: tuple[float, ...], ideal: np.ndarray) -> np.ndarray:
    """Apply the lens to undistorted pixel positions (inverse of ``ideal_pixels``)."""
    xy = (ideal.reshape(-1, 2) - k[:2, 2]) / k[0, 0]
    rays = np.c_[xy, np.ones(len(xy))]
    uv, _ = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), k, np.array(dist))
    return uv.reshape(ideal.shape)


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


def move_rig(rig: Rig, shift_mm: Vec3, turn_deg: float = 0.0) -> Rig:
    """Move every camera together: a turn about z, then a shift.

    The turn is about the vertical line through the mean of the targets, so a
    ring or dome spins around the point it looks at.

    Parameters
    ----------
    rig : Rig
        Cameras to move; unchanged.
    shift_mm : Vec3
        Translation applied to every eye and target, world mm.
    turn_deg : float
        Counter-clockwise turn seen from above (+z), degrees.

    Returns
    -------
    Rig
        Moved cameras; eye, target and up move rigidly, names and lenses kept.
    """
    a = math.radians(turn_deg)
    rot = np.array(
        [[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0, 0, 1]]
    )
    pivot = np.mean([c.target_mm for c in rig.cameras], axis=0) * (1, 1, 0)

    def move(p: Vec3) -> Vec3:
        return tuple(float(v) for v in rot @ (np.asarray(p) - pivot) + pivot + shift_mm)

    cameras = [
        Camera.model_validate(
            c.model_dump()
            | {
                "eye_mm": move(c.eye_mm),
                "target_mm": move(c.target_mm),
                "up": tuple(float(v) for v in rot @ c.up),
            }
        )
        for c in rig.cameras
    ]
    return Rig(cameras=cameras)


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
