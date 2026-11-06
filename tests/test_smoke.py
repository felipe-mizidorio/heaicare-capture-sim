from importlib.metadata import distributions

import heaicare_capture_sim


def test_package_imports() -> None:
    assert callable(heaicare_capture_sim.main)


def test_no_gui_opencv_alongside_headless() -> None:
    # CLAUDE.md: never install opencv-python next to opencv-python-headless (they clash on cv2).
    names = {d.metadata["Name"].lower() for d in distributions()}
    assert "opencv-python-headless" in names
    assert "opencv-python" not in names
