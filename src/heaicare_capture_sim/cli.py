"""Command line: capture, preset, gui."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from heaicare_capture_sim.capture import capture, require_empty_dir
from heaicare_capture_sim.head import load_mesh, make_synthetic_head
from heaicare_capture_sim.rig import dome, load_rig, ring, save_rig


def _cmd_capture(args: argparse.Namespace) -> int:
    rig = load_rig(args.rig)
    require_empty_dir(args.out)  # validate every input before starting Open3D
    head = load_mesh(args.mesh) if args.mesh else make_synthetic_head()
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


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="heaicare-capture-sim")
    sub = p.add_subparsers(required=True)

    c = sub.add_parser("capture", help="render every camera of a rig")
    c.add_argument("--rig", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--mesh", type=Path)
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
