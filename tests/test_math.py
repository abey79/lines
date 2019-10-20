import itertools

import numpy as np
import pytest

from lines.math import (
    vertices_matmul,
    segments_parallel_to_face,
    ParallelType,
    mask_segments,
    segments_outside_triangle_2d,
    split_segments,
    triangles_overlap_segment_2d,
    segment_triangle_intersection,
    NO_INT_PARALLEL_FRONT,
    NO_INT_PARALLEL_BEHIND,
    NO_INT_SHORT_SEGMENT_FRONT,
    NO_INT_SHORT_SEGMENT_BEHIND,
    NO_INT_OUTSIDE_FACE,
    INT_COPLANAR,
    INT_INSIDE,
    INT_EDGE,
    INT_VERTEX,
    DEGENERATE,
    mask_segment)
from lines.tables import CUBE_VERTICES, CUBE_SEGMENTS, CUBE_FACES


def test_vertices_matmul_empty():
    # Empty input is accepted, and returns empty output
    m = np.random.rand(3, 3)
    i = np.reshape(np.array([]), (0, 3))
    o = vertices_matmul(i, m)

    assert len(o) == 0
    assert np.all(o == i)


@pytest.mark.parametrize("n", (range(1, 5)))
def test_vertices_matmul_single_vertex(n):
    # a single vertex is accepted and is multiplied by a matrix
    m = np.random.rand(n, n)
    i = np.random.rand(n)
    o = vertices_matmul(i, m)
    assert np.all(m @ i == o)


@pytest.mark.parametrize("n", (range(1, 5)))
def test_vertices_matmul_single_vertex_array(n):
    # an array of a  single vertex is accepted and is multiplied by a matrix
    m = np.random.rand(n, n)
    i = np.random.rand(1, n)
    o = vertices_matmul(i, m)
    expected = m @ i.reshape((n,))
    assert np.all(expected == o)


def test_vertices_matmul_non_square_matrix():
    # non square matrices are not accepted and throw ValueError
    m = np.random.rand(3, 4)
    i = np.random.rand(1, 3)

    with pytest.raises(ValueError):
        vertices_matmul(i, m)


@pytest.mark.parametrize(("nm", "ni"), ((3, 4), (4, 3), (1, 5), (5, 1)))
def test_vertices_matmul_single_vertex_dim_mismatch(nm, ni):
    # when a single vertex is passed, mismatching dimension throw ValueError
    m = np.random.rand(nm, nm)
    i = np.random.rand(1, ni)
    with pytest.raises(ValueError):
        vertices_matmul(i, m)


def test_vertices_matmul_two_dimensions():
    segs = np.array(CUBE_VERTICES)
    m = np.random.rand(3, 3)
    o = vertices_matmul(segs, m)

    for i in range(len(segs)):
        assert np.all(o[i] == m @ segs[i])


def test_vertices_matmul_three_dimensions():
    vert = np.array(CUBE_VERTICES)
    s_idx = np.reshape(np.array(CUBE_SEGMENTS, dtype=np.uint32), (len(CUBE_SEGMENTS), 2))
    f_idx = np.reshape(np.array(CUBE_FACES, dtype=np.uint32), (len(CUBE_FACES), 3))
    segs = vert[s_idx]
    faces = vert[f_idx]

    m = np.random.rand(3, 3)
    o_segs = vertices_matmul(segs, m)
    o_faces = vertices_matmul(faces, m)

    for i in range(len(segs)):
        for j in range(2):
            assert np.all(o_segs[i][j] == m @ segs[i][j])

    for i in range(len(faces)):
        for j in range(3):
            assert np.all(o_faces[i][j] == m @ faces[i][j])


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


# Test data: array of tuple of test segment and expected result. If bool, result is
# expected regardless of accept_vertex_only value. If None, result is expected to be
# identical to accept_vertex_only value.
TRIANGLES_OVERLAP_SEGMENT_2D_TEST_DATA = [
    ([(-1, 10), (2, 10)], False),  # 0: no intersection at all
    ([(-1, -10), (2, -10)], False),
    ([(0.1, -0.0001), (0.9, -0.000001)], False),
    ([(0.51, 0.51), (1, 1)], False),
    ([(-0.5, 0.1), (-0.01, 0.8)], False),
    ([(-1, 0.5), (1, 0.5)], True),  # 5: regular intersection
    ([(0.5, -1), (0.5, 1)], True),
    ([(-0.5, 1), (1, -0.5)], True),
    ([(0, 2), (0, -0.1)], True),  # 8: 1 edge + 2 vertex overlap
    ([(-1, 0), (2, 0)], True),
    ([(-1, 2), (2, -1)], True),
    ([(0, 2), (0, 0.5)], True),  # 11: 1 edge + 1 vertex overlap
    ([(-1, 0), (0.5, 0)], True),
    ([(-1, 2), (0.5, 0.5)], True),
    ([(0, 0.8), (0, 0.5)], True),  # 14: 1 edge + 0 vertex overlap
    ([(0.1, 0), (0.5, 0)], True),
    ([(0.2, 0.8), (0.5, 0.5)], True),
    ([(0, 0), (0.1, 0.1)], True),  # 17: 1 vertex (superposed) + content overlap
    ([(1, 0), (0.1, 0.1)], True),
    ([(0, 1), (0.1, 0.1)], True),
    ([(-0.1, -0.1), (0.1, 0.1)], True),  # 20: 1 vertex + content overlap
    ([(2, -0.5), (-1, 1)], True),
    ([(-0.5, 2), (1, -1)], True),
    ([(0, 0), (0.6, 0.6)], True),  # 23: 1 vertex overlap + opposite edge crossing
    ([(1, 0), (-0.1, 0.5)], True),
    ([(0, 1), (0.5, -0.1)], True),
    ([(0, 0), (-3, -5)], None),  # 26: single vertex overlap (superposed)
    ([(1, 0), (2, -2)], None),
    ([(0, 1), (1, 2.5)], None),
    ([(-1, 0), (1, 2)], None),  # 29: single vertex overlap (not superposed)
    ([(-1, 1), (1, -1)], None),
    ([(0, -1), (2, 1)], None),
]


@pytest.mark.parametrize(("s", "expected"), TRIANGLES_OVERLAP_SEGMENT_2D_TEST_DATA)
def test_triangles_overlap_segment_2d(s, expected):
    triangles = np.array(list(itertools.permutations([(1, 0), (0, 1), (0, 0)])))

    for segment in [np.array(s), np.array([s[1], s[0]])]:
        if expected is None:
            assert np.all(triangles_overlap_segment_2d(triangles, segment, True))
            assert np.all(~triangles_overlap_segment_2d(triangles, segment, False))
        else:
            assert np.all(triangles_overlap_segment_2d(triangles, segment, True) == expected)
            assert np.all(triangles_overlap_segment_2d(triangles, segment, False) == expected)


SEGMENT_TRIANGLE_INTERSECTION_TEST_DATA = [
    # 0: parallel front
    ([(0.5, -0.5, 1), (0.5, 2, 1)], NO_INT_PARALLEL_FRONT, None, None, None, None),
    ([(0.5, 3, 1), (0.5, 2, 1)], NO_INT_PARALLEL_FRONT, None, None, None, None),
    ([(0, 0, 1), (0, 1, 1)], NO_INT_PARALLEL_FRONT, None, None, None, None),
    ([(0.1, 0.1, 1), (0.2, -0.2, 1)], NO_INT_PARALLEL_FRONT, None, None, None, None),
    ([(0, 0.1, 1), (0, 0.5, 1)], NO_INT_PARALLEL_FRONT, None, None, None, None),
    ([(0, 0, 1), (-1, -1, 1)], NO_INT_PARALLEL_FRONT, None, None, None, None),
    # 6: parallel behind
    ([(0.5, -0.5, -1), (0.5, 2, -1)], NO_INT_PARALLEL_BEHIND, None, None, None, None),
    ([(0.5, 3, -1), (0.5, 2, -1)], NO_INT_PARALLEL_BEHIND, None, None, None, None),
    ([(0, 0, -1), (0, 1, -1)], NO_INT_PARALLEL_BEHIND, None, None, None, None),
    ([(0.1, 0.1, -1), (0.2, -0.2, -1)], NO_INT_PARALLEL_BEHIND, None, None, None, None),
    ([(0, 0.1, -1), (0, 0.5, -1)], NO_INT_PARALLEL_BEHIND, None, None, None, None),
    ([(0, 0, -1), (-1, -1, -1)], NO_INT_PARALLEL_BEHIND, None, None, None, None),
    # 12: short segment front
    ([(0.1, 0.1, 0.1), (0.1, 0.1, 1)], NO_INT_SHORT_SEGMENT_FRONT, None, None, None, None),
    ([(0, 0, 0.1), (0, 0, 1)], NO_INT_SHORT_SEGMENT_FRONT, None, None, None, None),
    ([(-0.1, -0.1, 0.1), (-1, -1, 1)], NO_INT_SHORT_SEGMENT_FRONT, None, None, None, None),
    ([(0.5, 0, 0.1), (0.5, 0, 1)], NO_INT_SHORT_SEGMENT_FRONT, None, None, None, None),
    ([(0, 0.5, 0.1), (0, 0, 1)], NO_INT_SHORT_SEGMENT_FRONT, None, None, None, None),
    # 17: short segment behind
    ([(0.1, 0.1, -0.1), (0.1, 0.1, -1)], NO_INT_SHORT_SEGMENT_BEHIND, None, None, None, None),
    ([(0, 0, -0.1), (0, 0, -1)], NO_INT_SHORT_SEGMENT_BEHIND, None, None, None, None),
    ([(-0.1, -0.1, -0.1), (-1, -1, -1)], NO_INT_SHORT_SEGMENT_BEHIND, None, None, None, None),
    ([(0.5, 0, -0.1), (0.5, 0, -1)], NO_INT_SHORT_SEGMENT_BEHIND, None, None, None, None),
    ([(0, 0.5, -0.1), (0, 0, -1)], NO_INT_SHORT_SEGMENT_BEHIND, None, None, None, None),
    # 22: outside face
    ([(2, 2, -1), (2, 2, 1)], NO_INT_OUTSIDE_FACE, (2, 2, 0), 0.5, None, None),
    ([(2, 2, -1), (2, 2, 2)], NO_INT_OUTSIDE_FACE, (2, 2, 0), 1 / 3, None, None),
    ([(1, 1, 1.1), (-1, -1, -0.9)], NO_INT_OUTSIDE_FACE, (-0.1, -0.1, 0), 0.45, None, None),
    ([(0, 1, 1.1), (0, -1, -0.9)], NO_INT_OUTSIDE_FACE, (0, -0.1, 0), 0.45, None, None),
    # 26: coplanar
    ([(0.5, -0.5, 0), (0.5, 2, 0)], INT_COPLANAR, None, None, None, None),
    ([(0.5, 3, 0), (0.5, 2, 0)], INT_COPLANAR, None, None, None, None),
    ([(0, 0, 0), (0, 1, 0)], INT_COPLANAR, None, None, None, None),
    ([(0.1, 0.1, 0), (0.2, -0.2, 0)], INT_COPLANAR, None, None, None, None),
    ([(0, 0.1, 0), (0, 0.5, 0)], INT_COPLANAR, None, None, None, None),
    ([(0, 0, 0), (-1, -1, 0)], INT_COPLANAR, None, None, None, None),
    # 32: inside
    ([(0.5, 0.3, -1), (0.5, 0.3, 1)], INT_INSIDE, (0.5, 0.3, 0), 0.5, 0.5, 0.3),
    ([(-0.9, -0.9, -1), (3.1, 3.1, 3)], INT_INSIDE, (0.1, 0.1, 0), 0.25, 0.1, 0.1),
    # 34: edge
    ([(0.5, -1, -1), (0.5, 1, 1)], INT_EDGE, (0.5, 0, 0), 0.5, 0.5, 0),
    ([(1, 0, -1), (0, 0, 1)], INT_EDGE, (0.5, 0, 0), 0.5, 0.5, 0),
    ([(-1, 0.5, -1), (5, 0.5, 5)], INT_EDGE, (0, 0.5, 0), 1 / 6, 0, 0.5),
    # 37: vertex
    ([(0, 0, -1), (0, 0, 1)], INT_VERTEX, (0, 0, 0), 0.5, 0, 0),
    ([(1, 0, -1), (1, 0, 1)], INT_VERTEX, (1, 0, 0), 0.5, 1, 0),
    ([(0, 1, -1), (0, 1, 1)], INT_VERTEX, (0, 1, 0), 0.5, 0, 1),
    ([(-1, -1, -1), (1, 1, 1)], INT_VERTEX, (0, 0, 0), 0.5, 0, 0),
    ([(-1, 0, -1), (1, 0, 1)], INT_VERTEX, (0, 0, 0), 0.5, 0, 0),
    ([(1, 0, -1), (-1, 0, 1)], INT_VERTEX, (0, 0, 0), 0.5, 0, 0),
]


@pytest.mark.parametrize(
    ("seg", "result", "intersection", "expected_r", "expected_s", "expected_t"),
    SEGMENT_TRIANGLE_INTERSECTION_TEST_DATA,
)
def test_segment_triangle_intersection(
    seg, result, intersection, expected_r, expected_s, expected_t
):
    base_triangle = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    for triangle in itertools.permutations(base_triangle):
        for segment in [np.array(seg), np.array([seg[1], seg[0]])]:
            res, r, s, t, itrsct = segment_triangle_intersection(segment, np.array(triangle))

            assert res == result
            if intersection is not None:
                assert np.all(np.isclose(itrsct, np.array(intersection), atol=1e-14))
            if expected_r is not None:
                assert np.isclose(r, expected_r, atol=1e-14)

            # s and t are depend on triangle's vertex order
            if triangle == base_triangle:
                if expected_s is not None:
                    assert np.isclose(s, expected_s, atol=1e-14)
                if expected_t is not None:
                    assert np.isclose(t, expected_t, atol=1e-14)


def test_segment_triangle_intersection_degenerate():
    triangle = np.array([(0, 0, 0), (1, 1, 1), (2, 2, 2)])
    res, _, _, _, _ = segment_triangle_intersection(np.random.rand(2, 3), triangle)
    assert res == DEGENERATE


MASK_SEGMENT_TEST_DATA = [([(1, 0, -1), (0, 0, 1)], [(0.5, 0, 0), (0, 0, 1)])]


@pytest.mark.parametrize(("segment", "expected_masked_segment"), MASK_SEGMENT_TEST_DATA)
def test_mask_segment(segment, expected_masked_segment):
    triangle_array = np.array([[(0, 0, 0), (0, 1, 0), (1, 0, 0)]])

    results = mask_segment(np.array(segment), triangle_array)

    assert len(results) == 1
    assert results[0] == expected_masked_segment
