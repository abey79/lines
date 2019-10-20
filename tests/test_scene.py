import pytest

from lines import Scene, SegmentShape, TriangleShape


@pytest.fixture(params=["v1", "v2"])
def renderer_id(request):
    return request.param


def test_empty_scene(renderer_id):
    scene = Scene()
    scene.look_at((2, 2, 1), (0, 0, 1))
    scene.render(renderer_id)


def test_no_face(renderer_id):
    scene = Scene()
    scene.add(SegmentShape((0, -2, 1), (0, 1, 1)))
    scene.look_at((2, 2, 1), (0, 0, 1))
    scene.render(renderer_id)


def test_no_segment(renderer_id):
    scene = Scene()
    scene.add(TriangleShape((1, 0, 0), (-1, 0, 0), (0, 0, 2), add_segments=False))
    scene.look_at((2, 2, 1), (0, 0, 1))
    scene.render(renderer_id)
