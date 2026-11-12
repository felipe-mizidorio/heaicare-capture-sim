import json
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

from heaicare_capture_sim.cli import main
from heaicare_capture_sim.rig import ring, save_rig

RIG = ring(3, 300.0, 100.0, width=160, height=120)


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    """``serve`` in a subprocess: the test session already owns this process's renderer."""
    cwd = tmp_path_factory.mktemp("server")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "heaicare_capture_sim.cli",
            "serve",
            "--port",
            str(port),
        ],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.monotonic() + 60
    while True:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            break
        except OSError:
            if proc.poll() is not None or time.monotonic() > deadline:
                proc.kill()
                pytest.fail(f"server did not start:\n{proc.communicate()[0]}")
            time.sleep(0.2)
    yield f"http://127.0.0.1:{port}", cwd
    proc.send_signal(signal.SIGINT)
    out = proc.communicate(timeout=60)[0]
    # 0 = the renderer was released on its own thread (else Open3D aborts with 134).
    assert proc.returncode == 0, out


def _post(url, body, content_type="application/json"):
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    req = urllib.request.Request(url + "/capture", data, {"Content-Type": content_type})
    try:
        with urllib.request.urlopen(req, timeout=120) as res:
            return res.status, json.load(res)
    except urllib.error.HTTPError as exc:
        return exc.code, json.load(exc)


def test_invalid_rig_is_422(server):
    url, _ = server
    rig = RIG.model_dump(mode="json")
    rig["cameras"][1]["name"] = rig["cameras"][0]["name"]
    status, body = _post(url, {"rig": rig})
    assert status == 422 and "duplicate camera names" in json.dumps(body)


def test_bad_inputs_are_400(server, tmp_path):
    url, _ = server
    rig = RIG.model_dump(mode="json")
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "x").write_text("x")
    status, body = _post(url, {"rig": rig, "out": str(tmp_path / "full")})
    assert status == 400 and "not empty" in body["detail"]
    status, body = _post(url, {"rig": rig, "mesh": str(tmp_path / "nope.ply")})
    assert status == 400 and "nope.ply" in body["detail"]
    status, _ = _post(url, {"rig": rig, "scenario": "nowhere"})
    assert status == 400


def test_non_json_content_type_is_refused(server, tmp_path):
    # A web page can send a text/plain POST without a CORS preflight; it must not capture.
    url, _ = server
    out = tmp_path / "csrf"
    body = json.dumps({"rig": RIG.model_dump(mode="json"), "out": str(out)}).encode()
    status, _ = _post(url, body, content_type="text/plain")
    assert status == 422 and not out.exists()


def test_capture_matches_cli(server, tmp_path):
    url, _ = server
    status, body = _post(
        url, {"rig": RIG.model_dump(mode="json"), "out": str(tmp_path / "api")}
    )
    assert status == 200, body
    api = tmp_path / "api"
    assert body["out"] == str(api.resolve())
    assert body["cameras"] == json.loads((api / "cameras.json").read_text())

    save_rig(RIG, tmp_path / "rig.json")
    cli = subprocess.run(
        [
            sys.executable,
            "-m",
            "heaicare_capture_sim.cli",
            "capture",
            "--rig",
            str(tmp_path / "rig.json"),
            "--out",
            str(tmp_path / "cli"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert cli.returncode == 0, cli.stderr
    for cam in RIG.cameras:
        rel = f"images/{cam.name}.png"
        assert (api / rel).read_bytes() == (tmp_path / "cli" / rel).read_bytes()


def test_default_out_is_timestamped_capture_folder(server):
    url, cwd = server
    status, body = _post(url, {"rig": RIG.model_dump(mode="json")})
    assert status == 200, body
    out = cwd / "captures"
    assert [p.resolve() for p in out.iterdir()] == [(out / body["out"]).resolve()]
    assert len(list((out / body["out"] / "images").iterdir())) == 3


def test_serve_port_in_use_exits_2(capsys):
    with socket.create_server(("127.0.0.1", 0)) as s:
        assert main(["serve", "--port", str(s.getsockname()[1])]) == 2
    assert "error:" in capsys.readouterr().err
