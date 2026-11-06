"""Head meshes: procedural synthetic neonatal head and mesh loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

# Semi-axes (mm) of a neonatal-sized head: width x, length y (face +y), height z (crown +z).
SEMI_AXES_MM = (47.5, 60.0, 50.0)
_SKIN_RGB = np.array([0.87, 0.67, 0.58])


@dataclass(frozen=True)
class Head:
    """A renderable mesh and where it came from.

    Attributes
    ----------
    mesh : o3d.geometry.TriangleMesh
        Geometry in the world frame, mm.
    texture : np.ndarray or None
        Albedo map (H, W, 3) uint8 for the mesh UVs; None means use vertex colours.
    source : str
        "synthetic" or the mesh file path.
    """

    mesh: o3d.geometry.TriangleMesh
    texture: np.ndarray | None
    source: str


def _bilinear(grid: np.ndarray, h: int, w: int) -> np.ndarray:
    """Bilinear interpolation of a grid to a new size.

    Parameters
    ----------
    grid : np.ndarray
        Input grid to interpolate.
    h : int
        Output height.
    w : int
        Output width.

    Returns
    -------
    np.ndarray
        Interpolated grid of shape (h, w).
    """
    gy, gx = np.linspace(0, grid.shape[0] - 1, h), np.linspace(0, grid.shape[1] - 1, w)
    y0 = np.minimum(gy.astype(int), grid.shape[0] - 2)
    x0 = np.minimum(gx.astype(int), grid.shape[1] - 2)
    fy, fx = (gy - y0)[:, None], (gx - x0)[None, :]
    a, b = grid[np.ix_(y0, x0)], grid[np.ix_(y0, x0 + 1)]
    c, d = grid[np.ix_(y0 + 1, x0)], grid[np.ix_(y0 + 1, x0 + 1)]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def _skin_texture(seed: int, h: int = 1024, w: int = 2048) -> np.ndarray:
    """Multi-octave value noise on a skin tone, for feature matching.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility (default is 0).
    h : int, optional
        Texture height (default is 1024).
    w : int, optional
        Texture width (default is 2048).

    Returns
    -------
    np.ndarray
        Texture array of shape (h, w, 3) with uint8 dtype.
    """
    rng = np.random.default_rng(seed)
    acc = np.zeros((h, w))
    for cells, weight in ((8, 0.4), (32, 0.3), (128, 0.15), (512, 0.1)):
        acc += weight * _bilinear(rng.random((cells // 2 + 1, cells + 1)), h, w)
    acc += 0.05 * rng.random((h, w))
    rgb = _SKIN_RGB * (0.6 + 0.8 * acc)[..., None]
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


def make_synthetic_head(seed: int = 0) -> Head:
    """Ellipsoidal neonatal head with procedural skin texture, centred at the origin.

    Parameters
    ----------
    seed : int, optional
        Random seed for texture generation (default is 0).

    Returns
    -------
    Head
        A synthetic head with ellipsoidal mesh and procedural texture.
    """
    mesh = o3d.geometry.TriangleMesh.create_sphere(
        radius=1.0, resolution=120, create_uv_map=True
    )
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) * SEMI_AXES_MM)
    return Head(mesh=mesh, texture=_skin_texture(seed), source="synthetic")


def load_mesh(path: Path) -> Head:
    """Load a triangle mesh (PLY, OBJ, STL...) already in the world frame, mm.

    Parameters
    ----------
    path : Path
        Path to the mesh file.

    Returns
    -------
    Head
        A head with the loaded mesh.

    Raises
    ------
    FileNotFoundError
        If the mesh file does not exist.
    ValueError
        If the loaded mesh has no triangles.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"mesh not found: {path}")
    mesh = o3d.io.read_triangle_mesh(str(path))
    if len(mesh.triangles) == 0:
        raise ValueError(f"{path}: no triangles")
    texture = (
        np.asarray(mesh.textures[0])[..., :3].copy() if mesh.has_textures() else None
    )
    return Head(mesh=mesh, texture=texture, source=str(path))


def write_ground_truth(head: Head, path: Path) -> None:
    """Write geometry only (no colours/UVs), with UV-seam duplicates merged.

    Parameters
    ----------
    head : Head
        The head to write.
    path : Path
        Output file path.

    Raises
    ------
    AssertionError
        If the cleaned mesh is not watertight.
    """
    gt = o3d.geometry.TriangleMesh(head.mesh.vertices, head.mesh.triangles)
    gt.merge_close_vertices(1e-6)
    gt.remove_degenerate_triangles()
    gt.remove_duplicated_triangles()
    gt.remove_unreferenced_vertices()
    assert gt.is_watertight(), "Ground truth mesh is not watertight"
    o3d.io.write_triangle_mesh(str(path), gt)
