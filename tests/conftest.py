import pytest

from heaicare_capture_sim.renderer import Renderer


@pytest.fixture(scope="session")
def renderer() -> Renderer:
    """The one OffscreenRenderer allowed per process."""
    return Renderer(640, 480)
