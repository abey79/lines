import pytest

import numpy as np

from lines import Scene, SegmentShape, TriangleShape, Pyramid


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


# FIXME: use renderer_id
def test_pyramid():
    scene = Scene()
    scene.add(Pyramid())
    scene.look_at((2, 6, 1.5), (0, 0, 0))
    scene.perspective(70, 0.1, 10)
    rs = scene.render("v2", merge_lines=False)

    expected_idx = {1, 2, 5, 6, 7}

    idx = set()
    for ls in rs.mls:
        segment = np.array(ls)
        res = np.all(
            np.all(
                np.logical_or(
                    np.isclose(
                        rs._projected_segments[:, :, 0:2],
                        np.tile(segment.reshape((1, 2, 2)), (8, 1, 1)),
                    ),
                    np.isclose(
                        rs._projected_segments[:, [1, 0], 0:2],
                        np.tile(segment.reshape((1, 2, 2)), (8, 1, 1)),
                    ),
                ),
                axis=1,
            ),
            axis=1,
        )

        assert np.sum(res) == 1
        idx.add(np.argmax(res))
    assert idx == expected_idx
