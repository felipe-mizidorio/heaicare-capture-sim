import subprocess
import sys

from heaicare_capture_sim.cli import main
from heaicare_capture_sim.rig import load_rig, ring, save_rig


def test_preset_ring_and_dome(tmp_path):
    assert (
        main(
            [
                "preset",
                "ring",
                "--n",
                "6",
                "--radius-mm",
                "300",
                "--height-mm",
                "100",
                "--width",
                "640",
                "--height",
                "480",
                "-o",
                str(tmp_path / "r.json"),
            ]
        )
        == 0
    )
    rig = load_rig(tmp_path / "r.json")
    assert len(rig.cameras) == 6 and rig.cameras[0].width == 640
    assert (
        main(
            [
                "preset",
                "dome",
                "--n",
                "12",
                "--radius-mm",
                "300",
                "--min-elevation-deg",
                "20",
                "-o",
                str(tmp_path / "d.json"),
            ]
        )
        == 0
    )
    assert len(load_rig(tmp_path / "d.json").cameras) == 12


def test_bad_rig_files_exit_2(tmp_path, capsys):
    (tmp_path / "bad.json").write_text("{not json")
    for rig in (tmp_path / "bad.json", tmp_path / "missing.json"):
        assert main(["capture", "--rig", str(rig), "--out", str(tmp_path / "o")]) == 2
    err = capsys.readouterr().err
    assert "Traceback" not in err and err.count("error:") == 2


def test_non_empty_out_exit_2(tmp_path):
    save_rig(ring(2, 300.0, 100.0), tmp_path / "rig.json")
    (tmp_path / "o").mkdir()
    (tmp_path / "o" / "x").write_text("x")
    assert (
        main(
            [
                "capture",
                "--rig",
                str(tmp_path / "rig.json"),
                "--out",
                str(tmp_path / "o"),
            ]
        )
        == 2
    )


def test_unreadable_mesh_exit_2(tmp_path):
    save_rig(ring(2, 300.0, 100.0), tmp_path / "rig.json")
    assert (
        main(
            [
                "capture",
                "--rig",
                str(tmp_path / "rig.json"),
                "--out",
                str(tmp_path / "o"),
                "--mesh",
                str(tmp_path / "nope.ply"),
            ]
        )
        == 2
    )


def test_mesh_by_name_resolves_to_meshes_folder(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    save_rig(ring(2, 300.0, 100.0), tmp_path / "rig.json")
    args = ["capture", "--rig", "rig.json", "--out", "o", "--mesh", "nobody"]
    assert main(args) == 2
    assert "meshes/nobody" in capsys.readouterr().err


def test_capture_subprocess(tmp_path):
    # Subprocess: the test session already owns this process's one OffscreenRenderer.
    save_rig(ring(3, 300.0, 100.0, width=160, height=120), tmp_path / "rig.json")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "heaicare_capture_sim.cli",
            "capture",
            "--rig",
            str(tmp_path / "rig.json"),
            "--out",
            str(tmp_path / "run"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert len(list((tmp_path / "run" / "images").iterdir())) == 3
