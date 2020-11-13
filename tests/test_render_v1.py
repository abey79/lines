import numpy as np
import pytest

from .render_v1 import (
    ParallelType,
    mask_segments,
    segments_outside_triangle_2d,
    segments_parallel_to_face,
    split_segments,
)


def test_segments_outside_triangle_2d_basic():
    seg = np.array(
        [
            [(0.2, 0.2), (5, 5)],
            [(0.2, 0.2), (0.2, -5)],
            [(0.2, 0.2), (-5, 0.2)],
            [(0.2, 0.2), (0.3, 0.3)],
            [(0.2, -5), (0.8, -5)],
            [(10, 10), (10.1, 10.1)],
            [(-1, 5), (0.5, 10)],
            [(0.2, 5), (0.2, -5)],
        ]
    )

    assert np.all(
        np.equal(
            segments_outside_triangle_2d(seg, np.array([(0, 0), (1, 0), (0, 1)])),
            np.array([False, False, False, False, True, True, True, False]),
        )
    )


def test_segments_outside_triangle_2d_3d_input():
    seg3 = np.random.rand(50, 2, 3)
    tri3 = np.random.rand(3, 3)
    seg2 = np.array(seg3[:, :, 0:2])
    tri2 = np.array(tri3[:, 0:2])

    assert np.all(
        segments_outside_triangle_2d(seg3, tri3) == segments_outside_triangle_2d(seg2, tri2)
    )
    assert np.all(
        segments_outside_triangle_2d(seg3, tri2) == segments_outside_triangle_2d(seg2, tri3)
    )


def test_segments_parallel_to_face_single_segment():
    p0 = np.array([1, 2, 4])
    n = np.array([0, 0, 5])
    segments = np.array([[(100, 100, 100), (200, 300, 100)]])
    assert np.all(
        segments_parallel_to_face(segments, p0, n)
        == np.array([ParallelType.PARALLEL_FRONT.value])
    )


def test_segments_parallel_to_face_ground_plane():
    # ground plane
    p0 = np.array([0, 0, 0])
    n = np.array([0, 0, 3])

    segments = np.array(
        [
            [(0, 0, 0), (1, 0, 0)],  # coincident
            [(0, 0, 1), (1, 0, 1)],  # front
            [(0, 0, -1), (1, 0, -1)],  # back
            [(0, 0, -1), (1, 0, 1)],  # not parallel
        ]
    )

    assert np.all(
        segments_parallel_to_face(segments, p0, n)
        == np.array(
            [
                ParallelType.PARALLEL_COINCIDENT.value,
                ParallelType.PARALLEL_FRONT.value,
                ParallelType.PARALLEL_BACK.value,
                ParallelType.NOT_PARALLEL.value,
            ]
        )
    )


def test_segments_parallel_to_face_z_parallel_plane():
    with pytest.raises(ValueError):
        segments_parallel_to_face(
            np.random.rand(10, 2, 3), np.random.rand(3), np.array([3, 4, 0])
        )


@pytest.mark.parametrize(
    ("seg", "p0", "n"),
    [
        (np.random.rand(10, 2, 4), np.random.rand(3), np.array([1, 2, 3])),
        (np.random.rand(10, 2, 3), np.random.rand(4), np.array([1, 2, 3])),
        (np.random.rand(10, 2, 4), np.random.rand(3), np.array([1, 2, 3, 4])),
    ],
)
def test_segments_parallel_to_face_wrong_dimensions(seg, p0, n):
    with pytest.raises(ValueError):
        segments_parallel_to_face(seg, p0, n)


@pytest.mark.parametrize(
    ("seg", "mask"),
    [
        (np.random.rand(10, 2, 4), np.random.rand(5, 2)),
        (np.random.rand(10, 2, 3), np.random.rand(5, 3)),
        (np.random.rand(10, 3), np.random.rand(5, 2)),
        (np.random.rand(10, 2, 3), np.random.rand(5, 2, 2)),
    ],
)
def test_mask_segments_wrong_dimensions(seg, mask):
    with pytest.raises(ValueError):
        mask_segments(seg, mask)


def test_mask_segments_diff_then_intersect():
    segments = np.random.rand(500, 2, 3)
    mask = np.array([(0.1, 0.1), (0.2, 0.8), (0.8, 0.5)])

    seg_diff = mask_segments(segments, mask, True)
    assert len(seg_diff) > 0
    seg_diff_excl = mask_segments(seg_diff, mask, False)
    assert len(seg_diff_excl) == 0


def test_mask_segments_empty_results():
    pass


def test_mask_segments_empty_input():
    segments = np.array([]).reshape((0, 2, 3))
    mask = np.array([(0.5, -1), (0.5, 1), (10, 0)])

    assert np.all(mask_segments(segments, mask) == segments)


def test_mask_segment_single_segment():
    segments = np.array([[(0, 0, 0), (1, 0, 0)]])
    mask = np.array([(0.5, -1), (0.5, 1), (10, 0)])

    seg_diff = mask_segments(segments, mask, True)
    seg_excl = mask_segments(segments, mask, False)

    assert np.all(seg_diff == np.array([[(0, 0, 0), (0.5, 0, 0)]]))
    assert np.all(seg_excl == np.array([[(0.5, 0, 0), (1, 0, 0)]]))


def test_mask_segment_bug_disappearing_segment(root_directory):
    # Because of https://github.com/Toblerity/Shapely/issues/780, some z values are
    # randomly affected. This is test data extracted from a actual failing situation where
    # segments ended up missing because of the affected Z value.

    path = root_directory + "/tests/fixtures/"
    segments = np.load(path + "bug_segments.pickle", allow_pickle=True)
    face = np.load(path + "bug_face.pickle", allow_pickle=True)
    seg = np.load(path + "bug_seg_of_interest.pickle", allow_pickle=True)

    # the segment of interest "seg" should be present in "segments"
    assert np.any(np.all(np.all(segments == seg, axis=2), axis=1))

    results = mask_segments(segments, face, True)

    # the segment of interest should NOT be masked away with this particular mask
    assert np.any(np.all(np.all(results == seg, axis=2), axis=1))


def test_split_segments_basic():
    s0 = [0, 0, 0]
    s1 = [2, 2, 2]
    p0 = [0, 0, 1]
    n = [0, 0, 1]

    front, back = split_segments(np.array([[s0, s1]]), np.array(p0), np.array(n))

    assert np.all(back == np.array([[(s0, [1, 1, 1])]]))
    assert np.all(front == np.array([[([1, 1, 1], s1)]]))


def test_split_segments_opposite_n():
    s = np.random.rand(50, 2, 3)
    p0 = np.random.rand(3)
    n = np.random.rand(3)

    f1, b1 = split_segments(s, p0, n)
    f2, b2 = split_segments(s, p0, -n)

    assert np.all(f1 == f2)
    assert np.all(b1 == b2)
