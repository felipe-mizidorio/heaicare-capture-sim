"""Bring any OBJ into the world frame: units, axes, optional neck cut and rescale."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import open3d as o3d

from heaicare_capture_sim.head import Head, load_mesh

UNITS_MM = {"m": 1000.0, "dm": 100.0, "cm": 10.0, "mm": 1.0, "inch": 25.4}
AXES = {
    f"{sign}{name}": float(f"{sign}1") * np.eye(3)[i]
    for i, name in enumerate("xyz")
    for sign in "+-"
}
# MakeHuman exports a whole 1-year-old (its youngest) in dm, y up, face +z: cut at
# about a 1-year-old's head height and shrink to the synthetic head's length.
MAKEHUMAN = {
    "units": "dm",
    "up": "+y",
    "front": "+z",
    "neck_cut": 0.25,
    "head_length_mm": 120.0,
}


def to_world(up: str, front: str) -> np.ndarray:
    """Rotation taking the source's ``up`` to +z and ``front`` to +y.

    Parameters
    ----------
    up : str
        Source axis pointing to the crown, a key of ``AXES`` (e.g. "+y").
    front : str
        Source axis the face points along, perpendicular to ``up``.

    Returns
    -------
    np.ndarray
        3x3 proper rotation (det +1, never a mirror image).

    Raises
    ------
    ValueError
        If an axis is unknown or the two are not perpendicular.
    """
    for label, axis in (("up", up), ("front", front)):
        if axis not in AXES:
            raise ValueError(f"{label} must be one of {sorted(AXES)}, got {axis!r}")
    u, f = AXES[up], AXES[front]
    if u @ f != 0:
        raise ValueError(f"up {up} and front {front} must be perpendicular")
    return np.array([np.cross(f, u), f, u])  # rows: world x (right), y, z


def import_mesh(
    src: Path,
    dst: Path,
    units: str = "mm",
    up: str = "+z",
    front: str = "+y",
    neck_cut: float | None = None,
    head_length_mm: float | None = None,
    keep_body: bool = False,
) -> Head:
    """Put a mesh in the world frame and save it as a ready-to-use OBJ.

    Every part (material) is converted to mm and rotated so ``up`` is +z and
    ``front`` is +y. Optionally, triangles with a vertex more than
    ``neck_cut`` x height below the crown are dropped (or, with ``keep_body``,
    the skin below the cut is kept as a rendered-only part), and all parts are scaled
    uniformly so the head's front-to-back (y) extent is ``head_length_mm``.
    Finally the head's bounding box is centred on the origin. The head (ground
    truth) is the material with the most triangles; it is written as the first
    material, so ``load_mesh`` returns it as the head and the rest as extras.

    Parameters
    ----------
    src : Path
        Source OBJ (polygons are triangulated on load).
    dst : Path
        Output ``.obj`` (its folder is created); its ``.mtl`` and one texture
        PNG per part go next to it. Use ``meshes/<name>/<name>.obj`` so the GUI
        and ``capture --mesh <name>`` find it.
    units : str, optional
        Source units, a key of ``UNITS_MM`` (default "mm").
    up : str, optional
        Source axis toward the crown (default "+z", already world frame).
    front : str, optional
        Source axis the face points along (default "+y").
    neck_cut : float or None, optional
        Cut depth below the crown as a fraction of the source height, in (0, 1);
        None (default) keeps everything. For whole-body models.
    head_length_mm : float or None, optional
        Rescale to this front-to-back head extent; None (default) keeps the
        source size. For models that are not neonatal size.
    keep_body : bool, optional
        With ``neck_cut``, keep the head material below the cut as an extra
        part: rendered (it occludes like a real body) but not ground truth.
        Default False.

    Returns
    -------
    Head
        The converted head as ``load_mesh(dst)`` reads it.

    Raises
    ------
    FileNotFoundError
        If ``src`` does not exist.
    ValueError
        On bad arguments, an unreadable mesh, or no head left above the cut.
    OSError
        If ``dst`` cannot be written.
    """
    src, dst = Path(src), Path(dst)
    if units not in UNITS_MM:
        raise ValueError(f"units must be one of {sorted(UNITS_MM)}, got {units!r}")
    rotation = to_world(up, front)
    if neck_cut is not None and not 0.0 < neck_cut < 1.0:  # also rejects NaN
        raise ValueError(f"neck_cut must be in (0, 1), got {neck_cut}")
    if head_length_mm is not None and not (
        math.isfinite(head_length_mm) and head_length_mm > 0
    ):
        raise ValueError(f"head_length_mm must be positive, got {head_length_mm}")
    if keep_body and neck_cut is None:
        raise ValueError("keep_body needs neck_cut: the body is what lies below it")
    if dst.suffix.lower() != ".obj":
        raise ValueError(f"{dst}: output must be .obj to keep the textures")

    loaded = load_mesh(src)
    parts = [loaded, *loaded.extras]
    largest = max(range(len(parts)), key=lambda i: len(parts[i].mesh.triangles))
    head = parts.pop(largest)
    for part in (head, *parts):
        part.mesh.scale(UNITS_MM[units], center=np.zeros(3))
        part.mesh.rotate(rotation, center=np.zeros(3))

    if neck_cut is not None:
        z = np.asarray(head.mesh.vertices)[:, 2]
        cut = z.max() - neck_cut * (z.max() - z.min())
        body = o3d.geometry.TriangleMesh(head.mesh)
        if not _keep_above(head.mesh, cut):
            raise ValueError(f"{src}: no head left above the neck cut")
        parts = [p for p in parts if _keep_above(p.mesh, cut)]
        if keep_body and _keep_above(body, cut, above=False):
            parts.append(Head(body, head.texture, head.source))

    lo, hi = head.mesh.get_min_bound(), head.mesh.get_max_bound()
    scale = 1.0 if head_length_mm is None else head_length_mm / (hi[1] - lo[1])
    for part in (head, *parts):
        part.mesh.translate(-(lo + hi) / 2)
        part.mesh.scale(scale, center=np.zeros(3))

    dst.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(dst), _merge([head, *parts])):
        raise OSError(f"could not write mesh: {dst}")
    return load_mesh(dst)


def _keep_above(
    mesh: o3d.geometry.TriangleMesh, cut: float, above: bool = True
) -> bool:
    """Drop triangles with a vertex below z = ``cut``; False if none remain.

    With ``above=False`` keep exactly those triangles instead, so the two
    calls split a mesh without gap or overlap.

    Parameters
    ----------
    mesh : o3d.geometry.TriangleMesh
        Mesh to cut in place.
    cut : float
        Cut height, mm.
    above : bool, optional
        Keep the part above the cut (default True) or the rest.

    Returns
    -------
    bool
        Whether any triangle is left.
    """
    z = np.asarray(mesh.vertices)[:, 2]
    below = (z[np.asarray(mesh.triangles)] < cut).any(axis=1)
    mesh.remove_triangles_by_mask(below if above else ~below)
    mesh.remove_unreferenced_vertices()
    return len(mesh.triangles) > 0


def _merge(parts: list[Head]) -> o3d.geometry.TriangleMesh:
    """One mesh with one material per part, in order (the first is the head).

    Parameters
    ----------
    parts : list of Head
        Parts to merge.

    Returns
    -------
    o3d.geometry.TriangleMesh
        Merged mesh with textures and material ids set.
    """
    vertices, triangles, uvs, ids, offset = [], [], [], [], 0
    for i, part in enumerate(parts):
        v, t = np.asarray(part.mesh.vertices), np.asarray(part.mesh.triangles)
        vertices.append(v)
        triangles.append(t + offset)
        offset += len(v)
        uv = np.asarray(part.mesh.triangle_uvs)
        uvs.append(uv if len(uv) == 3 * len(t) else np.zeros((3 * len(t), 2)))
        ids.append(np.full(len(t), i, np.int32))
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.concatenate(vertices)),
        o3d.utility.Vector3iVector(np.concatenate(triangles)),
    )
    if all(p.mesh.has_vertex_colors() for p in parts):  # e.g. scanner colours
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.concatenate([np.asarray(p.mesh.vertex_colors) for p in parts])
        )
    if len(parts) == 1 and parts[0].texture is None:
        return mesh  # no materials needed; vertex colours (if any) render as-is
    mesh.triangle_uvs = o3d.utility.Vector2dVector(np.concatenate(uvs))
    mesh.triangle_material_ids = o3d.utility.IntVector(np.concatenate(ids))
    # An untextured part (e.g. MakeHuman's DefaultSkin) gets a white texel, so
    # each part stays its own material.
    white = np.full((1, 1, 3), 255, np.uint8)
    mesh.textures = [
        o3d.geometry.Image(p.texture if p.texture is not None else white) for p in parts
    ]
    return mesh
