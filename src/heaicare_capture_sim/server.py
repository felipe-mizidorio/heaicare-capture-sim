"""HTTP API: capture every camera of a posted rig, same engine as the CLI and GUI."""

from __future__ import annotations

import contextlib
import json
import socket
from collections.abc import AsyncIterator
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from heaicare_capture_sim.capture import (
    CAPTURE_ROOT,
    capture,
    new_capture_dir,
    require_empty_dir,
)
from heaicare_capture_sim.head import SYNTHETIC, load_head
from heaicare_capture_sim.renderer import Renderer, RenderWorker
from heaicare_capture_sim.rig import Rig
from heaicare_capture_sim.scenario import NONE, load_scenario

HOST = "127.0.0.1"  # no auth: never reachable from another machine


class CaptureRequest(BaseModel):
    """Body of ``POST /capture``; ``mesh`` and ``scenario`` mean what they do in the CLI."""

    model_config = ConfigDict(extra="forbid")

    rig: Rig
    mesh: str = SYNTHETIC
    scenario: str = NONE
    out: Path | None = None


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Same fixed viewport as the CLI and GUI, so all three give identical pixels.
    app.state.worker = RenderWorker(Renderer)
    try:
        yield
    finally:
        app.state.worker.close()


app = FastAPI(title="heaicare-capture-sim", lifespan=_lifespan)


@app.post("/capture")
def post_capture(body: CaptureRequest, request: Request) -> dict:
    """Render every camera; reply with the output folder and its ``cameras.json``.

    Runs in FastAPI's thread pool; captures queue on the one render thread.
    """
    out = None
    try:
        if body.out is not None:
            require_empty_dir(body.out)
        head = load_head(body.mesh)
        scenario = load_scenario(body.scenario)
        out = body.out if body.out is not None else new_capture_dir(CAPTURE_ROOT)
        request.app.state.worker.submit(
            lambda r: capture(body.rig, head, out, r, scenario)
        ).result()
    except Exception as exc:
        if body.out is None and out is not None:
            with contextlib.suppress(OSError):
                out.rmdir()  # the timestamped folder we made; no-op if it holds anything
        if isinstance(exc, (ValueError, OSError)):  # bad input: the CLI's exit code 2
            raise HTTPException(400, str(exc)) from exc
        raise
    return {
        "out": str(out.resolve()),
        "cameras": json.loads((out / "cameras.json").read_text()),
    }


def serve(sock: socket.socket) -> None:
    """Serve ``app`` on an already-bound socket until SIGINT/SIGTERM.

    Parameters
    ----------
    sock : socket.socket
        Listening socket on ``HOST``; binding first lets a taken port fail
        before Open3D starts.
    """
    host, port = sock.getsockname()[:2]
    print(f"API at http://{host}:{port} (docs at /docs)", flush=True)
    uvicorn.Server(uvicorn.Config(app, log_level="info")).run(sockets=[sock])
