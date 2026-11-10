"""Head meshes: procedural synthetic neonatal head and mesh loading."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

# Semi-axes (mm) of a neonatal-sized head: width x, length y (face +y), height z (crown +z).
SEMI_AXES_MM = (47.5, 60.0, 50.0)
_SKIN_RGB = np.array([0.87, 0.67, 0.58])
MESH_DIR = Path("meshes")  # one folder per head: meshes/<name>/<any>.obj
SYNTHETIC = "synthetic"


@dataclass(frozen=True)
class Head:
    """A renderable mesh and where it came from.

    Attributes
    ----------
    mesh : o3d.geometry.TriangleMesh
        Geometry in the world frame, mm. This is the ground-truth surface.
    texture : np.ndarray or None
        Albedo map (H, W, 3) uint8 for the mesh UVs, or (H, W, 4) for a
        see-through part; None means use vertex colours.
    source : str
        "synthetic" or the mesh file path.
    extras : tuple of Head
        Parts rendered with the head but left out of the ground truth
        (eyes, eyebrows, eyelashes, tongue).
    """

    mesh: o3d.geometry.TriangleMesh
    texture: np.ndarray | None
    source: str
    extras: tuple[Head, ...] = ()


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
    seed : int
        Random seed for reproducibility.
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


def _texture(mesh: o3d.geometry.TriangleMesh, k: int) -> np.ndarray | None:
    """Texture of material ``k`` as RGB, or RGBA if any texel is see-through.

    Parameters
    ----------
    mesh : o3d.geometry.TriangleMesh
        Mesh as read by Open3D.
    k : int
        Material id.

    Returns
    -------
    np.ndarray or None
        (H, W, 3) or (H, W, 4) uint8, or None if the material has no texture.
    """
    image = np.asarray(mesh.textures[k]) if k < len(mesh.textures) else None
    if image is None or image.ndim != 3:
        return None
    if image.shape[2] == 4 and (image[..., 3] < 255).any():
        return image.copy()
    return image[..., :3].copy()


def _select(
    mesh: o3d.geometry.TriangleMesh, keep: np.ndarray
) -> o3d.geometry.TriangleMesh:
    """Copy of ``mesh`` with only the ``keep`` triangles (UVs follow; materials dropped).

    Parameters
    ----------
    mesh : o3d.geometry.TriangleMesh
        Source mesh.
    keep : np.ndarray
        Boolean mask over triangles.

    Returns
    -------
    o3d.geometry.TriangleMesh
        The selected part.
    """
    part = o3d.geometry.TriangleMesh(mesh)
    part.remove_triangles_by_mask(~keep)
    part.remove_unreferenced_vertices()
    part.textures, part.triangle_material_ids = [], o3d.utility.IntVector()
    return part


def _extra(part: Head) -> Head | None:
    """Drop a see-through part's invisible triangles; make it opaque if it then is.

    MakeHuman's eye has a clear cornea shell mapped to alpha-0 texels. Drawn with
    the see-through shader, it shows the back of the eyeball; dropped, the
    eyeball renders opaque and exact like the skin.

    Parameters
    ----------
    part : Head
        An extra part.

    Returns
    -------
    Head or None
        The part to render, or None if none of it is visible.
    """
    texture = part.texture
    if texture is None or texture.shape[2] != 4 or not part.mesh.has_triangle_uvs():
        return part
    h, w = texture.shape[:2]
    uv = np.asarray(part.mesh.triangle_uvs)
    # In memory, Open3D textures have row 0 at v = 0 (see Renderer.set_scene).
    rows = np.clip((uv[:, 1] * h).astype(int), 0, h - 1)
    cols = np.clip((uv[:, 0] * w).astype(int), 0, w - 1)
    # ponytail: alpha sampled at triangle corners only; a triangle whose corners
    # are all opaque but whose interior is not renders opaque.
    alpha = texture[rows, cols, 3].reshape(-1, 3)
    visible = (alpha > 0).any(axis=1)
    if not visible.any():
        return None
    mesh = part.mesh if visible.all() else _select(part.mesh, visible)
    if (alpha[visible] == 255).all():
        texture = texture[..., :3].copy()
    return Head(mesh, texture, part.source)


def _read(path: Path) -> o3d.geometry.TriangleMesh:
    """Read a mesh file; OBJ polygons are fan-triangulated first.

    Open3D silently drops OBJ faces with more than three corners (MakeHuman and
    Blender write quads), so such files are rewritten to a temporary copy next
    to ``path``, where the MTL and texture paths still resolve.

    Parameters
    ----------
    path : Path
        Mesh file.

    Returns
    -------
    o3d.geometry.TriangleMesh
        The mesh as Open3D reads it.
    """
    if path.suffix.lower() != ".obj":
        return o3d.io.read_triangle_mesh(str(path))
    lines, polygons = [], False
    for line in path.read_text(errors="replace").splitlines():
        corners = line.split()[1:] if line.startswith("f ") else []
        if len(corners) > 3:
            # ponytail: fan triangulation, exact for convex polygons (modeller quads);
            # a concave polygon would need ear clipping.
            polygons = True
            lines += [
                f"f {corners[0]} {corners[i]} {corners[i + 1]}"
                for i in range(1, len(corners) - 1)
            ]
        else:
            lines.append(line)
    if not polygons:
        return o3d.io.read_triangle_mesh(str(path))
    fd, tmp = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".obj", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write("\n".join(lines) + "\n")
        return o3d.io.read_triangle_mesh(tmp)
    finally:
        Path(tmp).unlink()


def load_mesh(path: Path) -> Head:
    """Load a triangle mesh (PLY, OBJ, STL...) already in the world frame, mm.

    In a multi-material mesh the first material is the head (the ground-truth
    surface) and every other material becomes one of its ``extras``. Extras
    lose triangles that sample only fully transparent texels.

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
    mesh = _read(path)
    if len(mesh.triangles) == 0:
        raise ValueError(f"{path}: no triangles")
    # Open3D reads an OBJ as [empty default material, file materials...], so use
    # the material ids the triangles reference rather than textures[0].
    ids = np.asarray(mesh.triangle_material_ids, dtype=int)
    used = np.unique(ids) if len(ids) else np.array([0])
    parts = [
        Head(
            mesh if len(used) == 1 else _select(mesh, ids == k),
            _texture(mesh, k),
            str(path),
        )
        for k in used
    ]
    extras = tuple(e for e in map(_extra, parts[1:]) if e is not None)
    return Head(parts[0].mesh, parts[0].texture, str(path), extras)


def load_head(name: str | Path) -> Head:
    """The synthetic head, a head folder in ``MESH_DIR`` by name, or a mesh path.

    Parameters
    ----------
    name : str or Path
        "synthetic"; a bare name such as "baby_head" for the one ``.obj`` in
        ``MESH_DIR / "baby_head"`` (any file name); or any path (anything with a
        directory part or a suffix).

    Returns
    -------
    Head
        The loaded head; ``source`` is "synthetic" or the resolved path.

    Raises
    ------
    FileNotFoundError
        If the file or head folder does not exist.
    ValueError
        If a head folder does not hold exactly one ``.obj``, or the mesh has no
        triangles.
    """
    if str(name) == SYNTHETIC:
        return make_synthetic_head()
    path = Path(name)
    if path.parent == Path(".") and not path.suffix:
        folder = MESH_DIR / path
        if not folder.is_dir():
            raise FileNotFoundError(f"head folder not found: {folder}/")
        objs = sorted(folder.glob("*.obj"))
        if len(objs) != 1:
            raise ValueError(f"{folder}/ must hold one .obj, found {len(objs)}")
        path = objs[0]
    return load_mesh(path)


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
    OSError
        If the write operation fails.
    """
    gt = o3d.geometry.TriangleMesh(head.mesh.vertices, head.mesh.triangles)
    gt.merge_close_vertices(1e-6)
    gt.remove_degenerate_triangles()
    gt.remove_duplicated_triangles()
    gt.remove_unreferenced_vertices()
    if not o3d.io.write_triangle_mesh(str(path), gt):
        raise OSError(f"could not write ground truth: {path}")
