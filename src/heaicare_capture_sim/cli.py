"""Command line: capture, preset, gui, import-mesh."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from heaicare_capture_sim.capture import capture, require_empty_dir
from heaicare_capture_sim.head import SYNTHETIC, load_head
from heaicare_capture_sim.importer import AXES, MAKEHUMAN, UNITS_MM, import_mesh
from heaicare_capture_sim.rig import dome, load_rig, ring, save_rig


def _cmd_capture(args: argparse.Namespace) -> int:
    rig = load_rig(args.rig)
    require_empty_dir(args.out)  # validate every input before starting Open3D
    head = load_head(args.mesh)
    from heaicare_capture_sim.renderer import Renderer

    renderer = (
        Renderer()
    )  # same fixed viewport as the GUI, so both give identical pixels
    print(capture(rig, head, args.out, renderer))
    return 0


def _cmd_preset(args: argparse.Namespace) -> int:
    fields = {
        "width": args.width,
        "height": args.height,
        "focal_length_mm": args.focal_length_mm,
        "pixel_size_um": args.pixel_size_um,
    }
    target = tuple(args.target_mm)
    if args.layout == "ring":
        rig = ring(args.n, args.radius_mm, args.height_mm, target, **fields)
    else:
        rig = dome(args.n, args.radius_mm, args.min_elevation_deg, target, **fields)
    save_rig(rig, args.output)
    return 0


def _cmd_gui(args: argparse.Namespace) -> int:
    from heaicare_capture_sim.gui import run

    run(args.rig, args.mesh, args.port)
    return 0


_PRESETS = {"makehuman": MAKEHUMAN}
_IMPORT_OPTIONS = ("units", "up", "front", "neck_cut", "head_length_mm")


def _cmd_import_mesh(args: argparse.Namespace) -> int:
    options = dict(_PRESETS.get(args.preset, {}))
    options.update(
        {k: getattr(args, k) for k in _IMPORT_OPTIONS if getattr(args, k) is not None}
    )
    import_mesh(args.src, args.output, **options)
    print(args.output)
    return 0


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="heaicare-capture-sim")
    sub = p.add_subparsers(required=True)

    c = sub.add_parser("capture", help="render every camera of a rig")
    c.add_argument("--rig", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument(
        "--mesh",
        default=SYNTHETIC,
        help="'synthetic', a name in meshes/ (e.g. baby_head), or a mesh path",
    )
    c.set_defaults(func=_cmd_capture)

    pr = sub.add_parser("preset", help="generate a rig file")
    layouts = pr.add_subparsers(dest="layout", required=True)
    for name in ("ring", "dome"):
        lp = layouts.add_parser(name)
        lp.add_argument("--n", type=int, required=True)
        lp.add_argument("--radius-mm", type=float, required=True)
        if name == "ring":
            lp.add_argument("--height-mm", type=float, required=True)
        else:
            lp.add_argument("--min-elevation-deg", type=float, required=True)
        lp.add_argument("--target-mm", type=float, nargs=3, default=(0.0, 0.0, 0.0))
        lp.add_argument("--width", type=int, default=1920)
        lp.add_argument("--height", type=int, default=1080)
        lp.add_argument("--focal-length-mm", type=float, default=8.0)
        lp.add_argument("--pixel-size-um", type=float, default=3.45)
        lp.add_argument("-o", "--output", type=Path, required=True)
        lp.set_defaults(func=_cmd_preset)

    g = sub.add_parser("gui", help="interactive rig editor in the browser")
    g.add_argument("--rig", type=Path)
    g.add_argument("--mesh", type=Path)
    g.add_argument("--port", type=int, default=8080)
    g.set_defaults(func=_cmd_gui)

    m = sub.add_parser(
        "import-mesh", help="put any OBJ in the world frame as a head mesh"
    )
    m.add_argument("src", type=Path)
    m.add_argument("-o", "--output", type=Path, required=True)
    m.add_argument("--preset", choices=sorted(_PRESETS))
    m.add_argument("--units", choices=sorted(UNITS_MM))
    m.add_argument("--up", choices=sorted(AXES), help="source axis to the crown")
    m.add_argument(
        "--front", choices=sorted(AXES), help="source axis the face points along"
    )
    m.add_argument(
        "--neck-cut", type=float, help="cut depth below crown, fraction of height"
    )
    m.add_argument("--head-length-mm", type=float, help="rescale to this head length")
    m.set_defaults(func=_cmd_import_mesh)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns the exit code (2 = invalid input)."""
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        return args.func(args)
    except (
        ValueError,
        OSError,
    ) as exc:  # pydantic ValidationError subclasses ValueError
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
