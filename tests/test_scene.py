import pytest

from lines import Scene, SegmentShape, TriangleShape, Pyramid, StrippedCube


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


def test_pyramid(renderer_id):
    scene = Scene()
    scene.add(Pyramid())
    scene.look_at((2, 6, 1.5), (0, 0, 0))
    scene.perspective(70, 0.1, 10)
    rs = scene.render(renderer_id, merge_lines=False)

    assert rs.find_indices() == {1, 2, 5, 6, 7}


def test_city(renderer_id):
    """
    This test case appeared as problematic at some point
    """
    for i, j in [(0, 0), (-4, -2)]:
        scene = Scene()
        c = StrippedCube(scale=(1, 1, 1), translate=(i, j, 0.5))
        scene.add(c)
        scene.look_at((1.1, 0.8, 8.2), (0, 0.2, 0))
        scene.perspective(90, 0.1, 10)
        rs = scene.render(renderer_id, merge_lines=False)

        expected_indices = {8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}
        assert rs.find_indices() == expected_indices
