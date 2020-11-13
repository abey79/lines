import itertools
import pickle

import numpy as np
import pytest

# noinspection PyProtectedMember
from lines.math import (
    ATOL,
    DEGENERATE,
    INT_COPLANAR,
    INT_EDGE,
    INT_INSIDE,
    INT_VERTEX,
    NO_INT_OUTSIDE_FACE,
    NO_INT_PARALLEL_BEHIND,
    NO_INT_PARALLEL_FRONT,
    NO_INT_SHORT_SEGMENT_BEHIND,
    NO_INT_SHORT_SEGMENT_FRONT,
    _validate_shape,
    mask_segment,
    mask_segment_parallel,
    segment_triangle_intersection,
    segment_triangles_intersection,
    triangles_overlap_segment_2d,
    vertices_matmul,
)
from lines.tables import CUBE_FACES, CUBE_SEGMENTS, CUBE_VERTICES

from .utils import segment_list_equal

FACTOR_LIST = [1e6, 1e3, 1, 1e-3, 1e-6]


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


# Test data: array of tuple of test segment and expected result. If bool, result is
# expected regardless of accept_vertex_only value. If None, result is expected to be
# identical to accept_vertex_only value.
TRIANGLES_OVERLAP_SEGMENT_2D_TEST_DATA = [
    ([(-1, 10), (2, 10)], False),  # no intersection at all
    ([(-1, -10), (2, -10)], False),
    ([(0.1, -0.0001), (0.9, -0.000001)], False),
    ([(0.51, 0.51), (1, 1)], False),
    ([(-0.5, 0.1), (-0.01, 0.8)], False),
    ([(2, 0), (1.1, 0)], False),  # !! collinear
    ([(0, -1), (0, -0.1)], False),
    ([(2, -1), (1.5, -0.5)], False),
    ([(-1, 0.5), (1, 0.5)], True),  # regular intersection
    ([(0.5, -1), (0.5, 1)], True),
    ([(-0.5, 1), (1, -0.5)], True),
    ([(1, 0), (0, 0)], True),  # exactly one edge + 2 vertex overlap
    ([(1, 0), (0, 1)], True),
    ([(0, 0), (0, 1)], True),
    ([(0, 2), (0, -0.1)], True),  # 1 edge + 2 vertex overlap
    ([(2, 0), (0, 0)], True),
    ([(-1, 0), (2, 0)], True),
    ([(-1, 2), (2, -1)], True),
    ([(0.5, 0), (0, 0)], True),  # part of edge + 1 vertex overlap
    ([(0, 0.1), (0, 0)], True),
    ([(1, 0), (0.8, 0.2)], True),
    ([(0, 2), (0, 0.5)], True),  # 1 edge + 1 vertex overlap
    ([(-1, 0), (0.5, 0)], True),
    ([(-1, 2), (0.5, 0.5)], True),
    ([(0, 0.8), (0, 0.5)], True),  # 1 edge + 0 vertex overlap
    ([(0.1, 0), (0.5, 0)], True),
    ([(0.2, 0.8), (0.5, 0.5)], True),
    ([(0, 0), (0.1, 0.1)], True),  # 1 vertex (superposed) + content overlap
    ([(1, 0), (0.1, 0.1)], True),
    ([(0, 1), (0.1, 0.1)], True),
    ([(-0.1, -0.1), (0.1, 0.1)], True),  # 1 vertex + content overlap
    ([(2, -0.5), (-1, 1)], True),
    ([(-0.5, 2), (1, -1)], True),
    ([(0, 0), (0.6, 0.6)], True),  # 1 vertex overlap + opposite edge crossing
    ([(1, 0), (-0.1, 0.5)], True),
    ([(0, 1), (0.5, -0.1)], True),
    ([(0, 0), (-3, -5)], None),  # single vertex overlap (superposed)
    ([(1, 0), (2, -2)], None),
    ([(0, 1), (1, 2.5)], None),
    ([(2, 0), (1, 0)], None),  # !! collinear
    ([(-1, 0), (1, 2)], None),  # single vertex overlap (not superposed)
    ([(-1, 1), (1, -1)], None),
    ([(0, -1), (2, 1)], None),
]


@pytest.mark.parametrize("factor", FACTOR_LIST)
@pytest.mark.parametrize(("s", "expected"), TRIANGLES_OVERLAP_SEGMENT_2D_TEST_DATA)
def test_triangles_overlap_segment_2d(s, expected, factor):
    triangles = factor * np.array(list(itertools.permutations([(1, 0), (0, 1), (0, 0)])))

    for segment in [factor * np.array(s), factor * np.array([s[1], s[0]])]:
        if expected is None:
            assert np.all(triangles_overlap_segment_2d(triangles, segment, True))
            assert np.all(~triangles_overlap_segment_2d(triangles, segment, False))
        else:
            assert np.all(triangles_overlap_segment_2d(triangles, segment, True) == expected)
            assert np.all(triangles_overlap_segment_2d(triangles, segment, False) == expected)


def segment_triangles_intersection_wrapper(segment, triangle):
    """
    This function wraps ``segment_triangles_intersection()`` to make it compatible with
    ``segment_triangle_intersection()``.
    """
    res = 0
    r = 0.0
    s = 0.0
    t = 0.0
    b = 0.0
    itrsct = np.empty(3)

    # noinspection PyUnusedLocal
    def callback(seg, triangles, rres, rr, ss, tt, bb, iitrsct):
        assert len(triangles) == 1
        nonlocal res, r, s, t, b, itrsct
        res = rres
        r = rr
        s = ss
        t = tt
        b = bb
        itrsct = iitrsct

    segment_triangles_intersection(segment, np.array([triangle]), callback)
    return res, r, s, t, itrsct, b


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
    ([(1, 1, 1.1), (-1, -1, -0.9)], NO_INT_OUTSIDE_FACE, (-0.1, -0.1, 0), 0.55, None, None),
    ([(0, 1, 1.1), (0, -1, -0.9)], NO_INT_OUTSIDE_FACE, (0, -0.1, 0), 0.55, None, None),
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
    "func", (segment_triangle_intersection, segment_triangles_intersection_wrapper)
)
@pytest.mark.parametrize("factor", FACTOR_LIST)
@pytest.mark.parametrize(
    ("seg", "result", "intersection", "expected_r", "expected_s", "expected_t"),
    SEGMENT_TRIANGLE_INTERSECTION_TEST_DATA,
)
def test_segment_triangle_intersection(
    func, seg, result, intersection, expected_r, expected_s, expected_t, factor
):
    base_triangle = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    for triangle in itertools.permutations(base_triangle):
        for segment, invert_r in [(np.array(seg), False), (np.array([seg[1], seg[0]]), True)]:
            res, r, s, t, itrsct, _ = func(factor * segment, factor * np.array(triangle))

            assert res == result
            if intersection is not None:
                assert np.all(np.isclose(itrsct, factor * np.array(intersection), atol=1e-14))
            if expected_r is not None:
                assert np.isclose(
                    r, expected_r if not invert_r else 1 - expected_r, atol=1e-14
                )

            # s and t are depend on triangle's vertex order
            if triangle == base_triangle:
                if expected_s is not None:
                    assert np.isclose(s, expected_s, atol=1e-14)
                if expected_t is not None:
                    assert np.isclose(t, expected_t, atol=1e-14)


@pytest.mark.parametrize(
    "func", (segment_triangle_intersection, segment_triangles_intersection_wrapper)
)
def test_segment_triangle_intersection_degenerate(func):
    triangle = np.array([(0, 0, 0), (1, 1, 1), (2, 2, 2)])
    res, _, _, _, _, _ = func(np.random.rand(2, 3), triangle)
    assert res == DEGENERATE


@pytest.mark.parametrize(
    "func", (segment_triangle_intersection, segment_triangles_intersection_wrapper)
)
def test_segment_triangle_intersection_vertex_int_numerical_error(func, root_directory):
    """
    This is a particular case of encountered during debugging that highlighted numerical
    error issues
    """
    with open(
        root_directory + "/tests/fixtures/vertex_intersection_numerical_error.pickle", "rb"
    ) as f:
        m = pickle.load(f)
    res, _, _, _, itrsct, _ = func(m["segment"], m["face"])
    assert res == INT_VERTEX
    assert np.all(np.isclose(itrsct, m["segment"][1]))


def test_segment_list_equal():
    # quick and dirty test to make sure the test function works
    assert segment_list_equal(
        np.array([[(1, 2), (3, 4)], [(5, 6), (7, 8)]]),
        np.array([[(1, 2), (3, 4)], [(5, 6), (7, 8)]]),
    )

    assert segment_list_equal(
        np.array([[(1, 2), (3, 4)], [(5, 6), (7, 8)]]),
        np.array([[(5, 6), (7, 8)], [(1, 2), (3, 4)]]),
    )

    assert segment_list_equal(
        np.array([[(1, 2), (3, 4)], [(5, 6), (7, 8)]]),
        np.array([[(7, 8), (5, 6)], [(3, 4), (1, 2)]]),
    )


def check_mask_segment(func, segment, expected_masked_segment, triangle, factor):
    decimals = int(np.log10(factor) + np.log10(np.linalg.norm(segment)) + np.log10(ATOL))

    if expected_masked_segment is None:
        exp_res = np.array([segment])[:, :, 0:2]
    else:
        exp_res = np.array(expected_masked_segment)
    for face in itertools.permutations(triangle):
        for s in [segment, (segment[1], segment[0])]:
            results = func(
                factor * np.array(s, dtype=float), factor * np.array([face], dtype=float)
            )
            assert segment_list_equal(
                results[:, :, 0:2].round(decimals), (factor * exp_res).round(decimals)
            )


# if expected result is None, the test assumes it is identical to input
MASK_SEGMENT_TEST_DATA_FLAT_TRIANGLE = [
    # vertex interaction
    ([(-1, -1, -1), (0, 0, 0)], None),
    ([(-1, -1, 1), (0, 0, 0)], None),
    ([(0, -1, -1), (0, 0, 0)], None),
    ([(2, -1, 1), (1, 0, 0)], None),
    ([(1, 1, 1), (0, 0, 0)], None),
    ([(1, 1, -1), (0, 0, 0)], [[(1, 1), (0.5, 0.5)]]),
    ([(0.4, 0.4, -1), (0, 0, 0)], []),
    ([(0.3, 0.4, -1), (0, 0, 0)], []),
    ([(0.4, 0.3, 1), (0, 0, 0)], [[(0.4, 0.3), (0, 0)]]),
    # edge interaction
    ([(0, 0, 0), (1, 0, -1)], []),
    ([(0, 0, 0), (0.5, 0, -1)], []),
    ([(0, 0, 0), (2, 0, -2)], [[(1, 0), (2, 0)]]),
    ([(1, 0, -1), (0, 0, 1)], [[(0.5, 0), (0, 0)]]),
    ([(-1, 0, 1), (1, 0, -1)], [[(-1, 0), (0, 0)]]),
    ([(-1, 0, 1), (2, 0, -2)], [[(-1, 0), (0, 0)], [(1, 0), (2, 0)]]),
    ([(-1, 0, -1), (2, 0, 2)], None),
    ([(-1, 0, -1), (1, 0, 1)], None),
    ([(-1, 0, -1), (0.5, 0, 0.5)], None),
    ([(-1, 0.5, -1), (2, 0.5, 2)], None),
    ([(-1, 0.5, -1), (1, 0.5, 1)], None),
    ([(-1, 0.5, -1), (0.5, 0.5, 0.5)], None),
    ([(-1, 0.5, 1), (2, 0.5, -2)], [[(-1, 0.5), (0, 0.5)], [(0.5, 0.5), (2, 0.5)]]),
    ([(-1, 0.5, 1), (0.1, 0.5, -0.1)], [[(-1, 0.5), (0, 0.5)]]),
    ([(-1, 0.5, 1), (0.5, 0.5, -0.5)], [[(-1, 0.5), (0, 0.5)]]),
    # regular intersection
    ([(0.5, 0, 1), (0.5, 1, -1)], [[(0.5, 0), (0.5, 1)]]),
    ([(0.5, -1, 3), (0.5, 1, -1)], [[(0.5, -1), (0.5, 1)]]),
    ([(0.5, 0.1, 0.8), (0.5, 1, -1)], [[(0.5, 0.1), (0.5, 1)]]),
    ([(0.5, 0, 1), (0.5, 0.8, -1)], [[(0.5, 0), (0.5, 0.4)], [(0.5, 0.5), (0.5, 0.8)]]),
    ([(0, 0, -1), (1, 1, 3)], [[(0.25, 0.25), (1, 1)]]),
    # intersection outside
    ([(0.5, -1.5, -1), (0.5, 0.5, 1)], None),
    ([(0.5, -1.5, 1), (0.5, 0.5, -1)], [[(0.5, -1.5), (0.5, 0)]]),
    # short segment front
    ([(0.5, 0.5, 1), (0.5, 0.5, 2)], []),  # currently just a point
    ([(0.5, 0.5, 1), (0.4, 0.4, 2)], None),
    ([(0.2, 0.2, 2), (5, 5, 5)], [[(0.2, 0.2), (5, 5)]]),
    ([(2.5, 2.5, 1), (1.5, 3.5, 2)], None),
    # short segment back
    ([(0.5, 0.5, -1), (0.5, 0.5, -2)], []),  # currently just a point
    ([(0.5, 0.5, -1), (0.4, 0.4, -2)], []),
    ([(0.2, 0.2, -2), (5, 5, -5)], [[(0.5, 0.5), (5, 5)]]),
    ([(2.5, 2.5, -1), (1.5, 3.5, -2)], None),
]


@pytest.mark.parametrize("func", (mask_segment, mask_segment_parallel))
@pytest.mark.parametrize("factor", FACTOR_LIST)
@pytest.mark.parametrize(
    ("segment", "expected_masked_segment"), MASK_SEGMENT_TEST_DATA_FLAT_TRIANGLE
)
def test_mask_segment_flat_triangle(func, segment, expected_masked_segment, factor):
    check_mask_segment(
        func, segment, expected_masked_segment, [(0, 0, 0), (0, 1, 0), (1, 0, 0)], factor
    )


# if expected result is None, the test assumes it is identical to input
MASK_SEGMENT_TEST_DATA_TILTED_TRIANGLE = [
    # Intersection outside
    ([(-1, -1, -1), (0, 0, 0)], None),
    ([(-1, -1, 1), (0, 0, 0)], None),
    # vertex interaction
    ([(0, 0, 1), (-1, 0, 0.9)], None),
    ([(0, 0, 1), (-1, 0, 1)], None),
    ([(0, 0, 1), (-1, 0, 1.1)], None),
    ([(0, 0, 1), (1, 0, 0.1)], None),
    ([(0, 0, 1), (1, 0, 0.9)], None),
    ([(0, 0, 1), (1, 0, 1)], None),
    ([(0, 0, 1), (1, 0, 1.1)], None),
    ([(0, 0, 1), (0.9, 0, 0)], []),
    ([(0, 0, 1), (0.1, 0, 0)], []),
    ([(0, 0, 1), (2, 0, -1.1)], [[(1, 0), (2, 0)]]),
    # edge interaction
    ([(1.1, 0.1, -1), (0.9, 0.1, 1)], None),
    ([(2, 0, -1), (0, 0, 0.5)], [[(2, 0), (1, 0)]]),
    ([(0.9, 0.1, -1), (1.1, 0.1, 1)], [[(1, 0.1), (1.1, 0.1)]]),
    ([(-0.5, 0, -2), (1.5, 0, 1)], [[(-0.5, 0), (0, 0)], [(1.5, 0), (1, 0)]]),
]


@pytest.mark.parametrize("func", (mask_segment, mask_segment_parallel))
@pytest.mark.parametrize("factor", FACTOR_LIST)
@pytest.mark.parametrize(
    ("segment", "expected_masked_segment"), MASK_SEGMENT_TEST_DATA_TILTED_TRIANGLE
)
def test_mask_segment_tilted_triangle(func, segment, expected_masked_segment, factor):
    check_mask_segment(
        func, segment, expected_masked_segment, [(1, 1, 0), (1, -1, 0), (0, 0, 1)], factor
    )


# In this test we consider only segment parallel to XY face, so the test can generate z
# coordinates that are either below, on, or above the plane. Expected result pertains only to
# the z < 0 case.
MASK_SEGMENT_SEG_PARALLEL_TO_FACE_TEST_DATA = [
    # no edge interaction
    ([(0.5, -1), (0.5, 2)], [[(0.5, -1), (0.5, 0)], [(0.5, 0.5), (0.5, 2)]]),
    ([(0.5, -1), (0.5, 0.1)], [[(0.5, -1), (0.5, 0)]]),
    ([(0.5, 0.1), (0.5, 0.3)], []),
    ([(2, -1), (2, 2)], [[(2, -1), (2, 2)]]),
    # edge interaction
    ([(0, 0), (0, 1)], []),
    ([(1, 0), (0, 1)], []),
    ([(1, 0), (0, 0)], []),
    ([(0.8, 0), (0.2, 0)], []),
    ([(2, 0), (0.2, 0)], [[(2, 0), (1, 0)]]),
    ([(2, 0), (0, 0)], [[(2, 0), (1, 0)]]),
    ([(2, -1), (-1, 2)], [[(2, -1), (1, 0)], [(0, 1), (-1, 2)]]),
    # vertex interaction
    ([(1, 0), (3, 0)], [[(1, 0), (3, 0)]]),
    ([(1, 0), (3, 0.1)], [[(1, 0), (3, 0.1)]]),
    ([(0, 0), (-1, 0)], [[(0, 0), (-1, 0)]]),
    ([(0, 0), (0, -1)], [[(0, 0), (0, -1)]]),
    ([(0, 0), (-1, -1)], [[(0, 0), (-1, -1)]]),
    ([(-1, 1), (1, -1)], [[(-1, 1), (0, 0)], [(0, 0), (1, -1)]]),
    ([(-0.5, 1), (0.5, -1)], [[(-0.5, 1), (0, 0)], [(0, 0), (0.5, -1)]]),
    ([(1, 0), (0.1, 0.1)], []),
    ([(0, 0), (0.5, 0.5)], []),
    ([(0, 0), (0.7, 0.3)], []),
]


@pytest.mark.parametrize("func", (mask_segment, mask_segment_parallel))
@pytest.mark.parametrize(
    ("segment", "expected_masked_segment"), MASK_SEGMENT_SEG_PARALLEL_TO_FACE_TEST_DATA
)
def test_mask_segment_seg_parallel_to_face(func, segment, expected_masked_segment):
    # We add z coordinates to the segment. In case of z = -1, the segment should be masked
    # and the results as expected. Otherwise the segment shouldn't be masked and the result
    # identical to input.
    for triangle in itertools.permutations([(0, 0, 0), (0, 1, 0), (1, 0, 0)]):
        for s in [segment, (segment[1], segment[0])]:
            for z in [-1, 0, 1]:
                seg = np.hstack((np.array(s, dtype=float), z * np.ones(shape=(2, 1))))

                results = func(seg, np.array([triangle]))

                if z == -1:
                    assert segment_list_equal(
                        np.array(expected_masked_segment), results[:, :, 0:2]
                    )
                else:
                    assert segment_list_equal(np.array([s]), results[:, :, 0:2])


def test_mask_segment_seg_parallel_to_cam_plane():
    # TODO
    # case where segment is parallel to cam plane and face is not
    pass


def segment_triangles_intersection_validation_callback(
    segment, triangles, res, r, s, t, b, itrsct
):
    assert _validate_shape(segment, 2, 3)
    assert _validate_shape(triangles, None, 3, 3)
    assert len(triangles) > 0

    for i, triangle in enumerate(triangles):
        if i == 0 and res == 7:
            print("hello")

        e_res, e_r, e_s, e_t, e_itrsct, e_b = segment_triangle_intersection(segment, triangle)

        assert np.isclose(e_res, res, rtol=1e-10)

        if e_res in [NO_INT_SHORT_SEGMENT_FRONT, NO_INT_SHORT_SEGMENT_BEHIND]:
            assert np.isclose(e_r, r[i], rtol=1e-10)

        if e_res in [NO_INT_OUTSIDE_FACE, INT_VERTEX, INT_EDGE, INT_INSIDE]:
            assert np.isclose(e_r, r[i], rtol=1e-10)
            if e_res != NO_INT_OUTSIDE_FACE:
                assert np.isclose(e_s, s[i], rtol=1e-10)
                assert np.isclose(e_t, t[i], rtol=1e-10)
            assert np.isclose(e_b, b[i], rtol=1e-10)
            assert np.all(np.isclose(e_itrsct, itrsct[i]))


def test_segment_triangles_intersection_random():
    k = 10
    n = 100
    segments = np.random.rand(k, 2, 3)
    triangles = np.random.rand(n, 3, 3)

    for segment in segments:
        segment_triangles_intersection(
            segment, triangles, segment_triangles_intersection_validation_callback
        )


def test_segment_triangles_intersection_quantized_random():
    # We apply strong quantization on random value to promote the emergence of degenerate and
    # parallel segment/face pairs
    k = 5
    n = 1000
    segments = np.around(5 * np.random.rand(k, 2, 3))
    triangles = np.around(5 * np.random.rand(n, 3, 3))

    for segment in segments:
        segment_triangles_intersection(
            segment, triangles, segment_triangles_intersection_validation_callback
        )


def test_mask_segment_random():
    k = 5
    n = 200
    segments = np.random.rand(k, 2, 3)
    triangles = np.random.rand(n, 3, 3)

    for segment in segments:
        res1 = mask_segment(segment, triangles)
        res2 = mask_segment_parallel(segment, triangles)

        assert segment_list_equal(res1, res2)


@pytest.mark.skip("FIXME: failes for i == 7118")
def test_mask_segment_quantized_random():
    # TODO: this test fails in a specific case!
    k = 1
    n = 5
    np.random.seed(12)

    for i in [7118]:  # range(10000):
        np.random.seed(i)
        segments = np.around(5 * np.random.rand(k, 2, 3))
        triangles = np.around(5 * np.random.rand(n, 3, 3))

        for segment in segments:
            res1 = mask_segment(segment, triangles)
            res2 = mask_segment_parallel(segment, triangles)

            assert segment_list_equal(
                res1[:, :, 0:2].round(decimals=12), res2[:, :, 0:2].round(decimals=12)
            )
