import collections
import enum
import logging

import numpy as np
import shapely.ops
from shapely.geometry import Polygon, asLineString

RTOL = 1e-14
ATOL = 1e-14


def _validate_segments(segments: np.ndarray) -> None:
    if len(segments.shape) != 3 or segments.shape[1:] != (2, 3):
        raise ValueError(f"segments array has shape {segments.shape} instead of (N, 2, 3)")


def _validate_shape(a: np.ndarray, *args):
    if len(a.shape) != len(args):
        return False

    for s, v in zip(a.shape, args):
        if isinstance(v, int):
            if s != v:
                return False
        elif isinstance(v, collections.abc.Iterable):
            if s not in v:
                return False
        else:
            if v is not None:
                return False

    return True


def vertices_matmul(vertices: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Apply a matrix multiplication to all vertices in the input array. The vertices coordinates
    are assumed to be stored in the input's last dimension.
    :param vertices: [d0 x ... x dN x M] N-dimensional array of M-sized vertices
    :param matrix: [M x M] matrix
    :return: transformed vertices (identical shape as input)
    """

    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")

    if vertices.shape[-1] != matrix.shape[0]:
        raise ValueError(
            f"matrix dimension ({matrix.shape[0]}x{matrix.shape[1]}) does not match vertex "
            f"dimension ({vertices.shape[-1]})"
        )

    if len(vertices) == 0:
        return np.empty_like(vertices)

    if len(vertices.shape) == 1:
        return matrix @ vertices

    # vertices needs to be reshaped such that the last two dimensions are (..., N, 1)
    # then matmul can be applied as it broadcast the matrix on the last two dimension of the
    # other operand
    shape = vertices.shape
    output = matrix @ vertices.reshape((*shape, 1))
    output.shape = shape
    return output


# noinspection DuplicatedCode
def segments_outside_triangle_2d(segments: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    """
    Compute which segments are outside the face, considering only 2D projection along Z axis.
    As a results, the input's Z data is disregarded and optional.
    :param segments: (M x 2 x 2-3) segments
    :param triangle: (3 x 2-3) face
    :return: Mx1 array of boolean, true for segments outside of the triangle
    """

    if len(segments.shape) != 3 or segments.shape[1] != 2 or not segments.shape[2] in (2, 3):
        raise ValueError(
            f"segments array has shape {segments.shape} instead of (N, 2, 2 or 3)"
        )

    if len(triangle.shape) != 2 or not triangle.shape[1] in (2, 3):
        raise ValueError(f"triangle array has shape {triangle.shape} instead of (3, 2 or 3)")

    """
    https://gamedev.stackexchange.com/a/21110
    if t0, t1 and t2 are all on the same side of line P0P1, return NOT INTERSECTING
    if P0 AND P1 are on the other side of line t0t1 as t2, return NOT INTERSECTING
    if P0 AND P1 are on the other side of line t1t2 as t0, return NOT INTERSECTING
    if P0 AND P1 are on the other side of line t2t0 as t1, return NOT INTERSECTING
    """

    p0 = segments[:, 0, 0:2]
    p1 = segments[:, 1, 0:2]
    t0 = triangle[0, 0:2]
    t1 = triangle[1, 0:2]
    t2 = triangle[2, 0:2]

    p0p1 = p1 - p0

    p0t0 = t0 - p0
    p0t1 = t1 - p0
    p0t2 = t2 - p0

    t0p0 = -p0t0
    t1p0 = -p0t1
    t2p0 = -p0t2

    t0p1 = p1 - t0
    t1p1 = p1 - t1
    t2p1 = p1 - t2

    t0t1 = t1 - t0
    t1t2 = t2 - t1
    t2t0 = t0 - t2

    t1t0 = -t0t1
    t2t1 = -t1t2
    t0t2 = -t2t0

    f1 = np.cross(t0t1, t0p0) * np.cross(t0t1, t0t2)
    f2 = np.cross(t0t1, t0t2) * np.cross(t0t1, t0p1)

    f3 = np.cross(t1t2, t1p0) * np.cross(t1t2, t1t0)
    f4 = np.cross(t1t2, t1t0) * np.cross(t1t2, t1p1)

    f5 = np.cross(t2t0, t2p0) * np.cross(t2t0, t2t1)
    f6 = np.cross(t2t0, t2t1) * np.cross(t2t0, t2p1)

    p0p1_cross_p0t1 = np.cross(p0p1, p0t1)
    f7 = np.cross(p0p1, p0t0) * p0p1_cross_p0t1
    f8 = p0p1_cross_p0t1 * np.cross(p0p1, p0t2)

    # some tolerance is accepted to ensure that face do not hide segment running behind
    # along the edge, as it regularly happens with SilhouetteSkin
    return (
        np.logical_and(f1 <= 1e-12, f2 <= 1e-12)
        | np.logical_and(f3 <= 1e-12, f4 <= 1e-12)
        | np.logical_and(f5 <= 1e-12, f6 <= 1e-12)
        | np.logical_and(f7 >= 1e-12, f8 >= 1e-12)
    )


# noinspection DuplicatedCode
def triangles_overlap_segment_2d(
    triangles: np.ndarray, segment: np.ndarray, accept_vertex_only: bool = True
) -> np.ndarray:
    """
    Compute which triangles overlap a segment, considering only 2D projection along Z axis.
    The input's Z data is disregarded and optional.

    .. note::
        This function's implementation is heavily inspired by this `GameDev StackExchange
        question <https://gamedev.stackexchange.com/a/21110>`_

    :param triangles: (M x 3 x 2-3) triangles
    :param segment: (2 x 2-3) segment
    :param accept_vertex_only: if False, vertex only intersection are not accepted
    :return: Mx1 array of boolean, true faces overlapping the segment
    """

    if not _validate_shape(triangles, None, 3, (2, 3)):
        raise ValueError(f"triangles array has shape {triangles.shape} instead of (N, 3, 2|3)")

    if not (
        _validate_shape(segment, 2, (2, 3))
        or _validate_shape(segment, len(triangles), 2, (2, 3))
    ):
        raise ValueError(
            f"segment array has shape {segment.shape} instead of (2, 2|3) or (M, 2, 2|3)"
        )

    if len(segment.shape) == 2:
        p0 = segment[0, 0:2]
        p1 = segment[1, 0:2]
    else:
        p0 = segment[:, 0, 0:2]
        p1 = segment[:, 1, 0:2]

    t0 = triangles[:, 0, 0:2]
    t1 = triangles[:, 1, 0:2]
    t2 = triangles[:, 2, 0:2]

    p0p1 = p1 - p0

    p0t0 = t0 - p0
    p0t1 = t1 - p0
    p0t2 = t2 - p0

    t0p0 = -p0t0
    t1p0 = -p0t1
    t2p0 = -p0t2

    t0p1 = p1 - t0
    t1p1 = p1 - t1
    t2p1 = p1 - t2

    t0t1 = t1 - t0
    t1t2 = t2 - t1
    t2t0 = t0 - t2

    t1t0 = -t0t1
    t2t1 = -t1t2
    t0t2 = -t2t0

    p0p1_cross_p0t0 = np.cross(p0p1, p0t0)
    p0p1_cross_p0t1 = np.cross(p0p1, p0t1)
    p0p1_cross_p0t2 = np.cross(p0p1, p0t2)

    f = np.stack(
        (
            np.cross(t0t1, t0p0) * np.cross(t0t1, t0t2),
            np.cross(t0t1, t0t2) * np.cross(t0t1, t0p1),
            np.cross(t1t2, t1p0) * np.cross(t1t2, t1t0),
            np.cross(t1t2, t1t0) * np.cross(t1t2, t1p1),
            np.cross(t2t0, t2p0) * np.cross(t2t0, t2t1),
            np.cross(t2t0, t2t1) * np.cross(t2t0, t2p1),
            p0p1_cross_p0t0 * p0p1_cross_p0t1,
            p0p1_cross_p0t1 * p0p1_cross_p0t2,
            p0p1_cross_p0t2 * p0p1_cross_p0t0,
        ),
        axis=0,
    )

    # Here we need to carefully compute the tolerance. All the f-numbers are products of 2
    # cross-products, so in the order of the specific length to the power of 4.
    f_atol = 1e-12 * (np.linalg.norm(p0p1) ** 4)
    f = np.where(np.isclose(f, 0, atol=f_atol), 0, f)

    not_intersecting = (
        np.logical_and(f[0] < 0, f[1] < 0)
        | np.logical_and(f[2] < 0, f[3] < 0)
        | np.logical_and(f[4] < 0, f[5] < 0)
        | np.logical_and(np.logical_and(f[6] > 0, f[7] > 0), f[8] > 0)
    )

    if accept_vertex_only:
        return ~not_intersecting
    else:
        fsgn = np.sign(f[0:6])

        # This is False in cases a single vertex is touching
        overlapping = (
            np.logical_and(
                np.logical_and(f[0] == 0, f[1] == 0),
                ~np.logical_or(
                    np.logical_and(
                        np.sum(fsgn[2:4], axis=0) == 2, np.sum(fsgn[4:6], axis=0) == -1
                    ),
                    np.logical_and(
                        np.sum(fsgn[2:4], axis=0) == -1, np.sum(fsgn[4:6], axis=0) == 2
                    ),
                ),
            )
            | np.logical_and(
                np.logical_and(f[2] == 0, f[3] == 0),
                ~np.logical_or(
                    np.logical_and(
                        np.sum(fsgn[0:2], axis=0) == 2, np.sum(fsgn[4:6], axis=0) == -1
                    ),
                    np.logical_and(
                        np.sum(fsgn[0:2], axis=0) == -1, np.sum(fsgn[4:6], axis=0) == 2
                    ),
                ),
            )
            | np.logical_and(
                np.logical_and(f[4] == 0, f[5] == 0),
                ~np.logical_or(
                    np.logical_and(
                        np.sum(fsgn[0:2], axis=0) == 2, np.sum(fsgn[2:4], axis=0) == -1
                    ),
                    np.logical_and(
                        np.sum(fsgn[0:2], axis=0) == -1, np.sum(fsgn[2:4], axis=0) == 2
                    ),
                ),
            )
        )

        # this is True even when a single vertex is touched
        touching = (
            np.logical_and(f[0] <= 0, f[1] <= 0)
            | np.logical_and(f[2] <= 0, f[3] <= 0)
            | np.logical_and(f[4] <= 0, f[5] <= 0)
            | np.logical_and(np.logical_and(f[6] >= 0, f[7] >= 0), f[8] >= 0)
        )

        return np.logical_and(~not_intersecting, np.logical_or(~touching, overlapping))


class ParallelType(enum.Enum):
    """
    Enum to describe parallelism information.
    Front/back axis is z, and front means higher z value
    """

    NOT_PARALLEL = 0
    PARALLEL_COINCIDENT = 1
    PARALLEL_BACK = 2
    PARALLEL_FRONT = 3


def segments_parallel_to_face(
    segments: np.ndarray, p0: np.ndarray, n: np.ndarray
) -> np.ndarray:
    """
    Compare an array of segments to a plane defined by a point and a normal, and return an
    array containing ParallelType information for each segment. The plan must not be parallel
    to the z axis.
    :param segments: Mx2x3 array of segment
    :param p0: plane reference point (length 3)
    :param n: plane normal (length 3, does not need to be normalized)
    :return: length M array of ParallelType
    """

    _validate_segments(segments)

    if p0.shape != (3,) or n.shape != (3,):
        raise ValueError(f"p0 and n must be of length 3")

    if np.isclose(n[2], 0, atol=ATOL, rtol=RTOL):
        raise ValueError(f"plane should not be parallel to z axis")

    # make sure n points up
    if n[2] < 0:
        n = -n

    cnt = len(segments)
    s0 = segments[:, 0]
    sv = np.diff(segments, axis=1).reshape(cnt, 3)

    output = np.empty(cnt)
    output.fill(ParallelType.NOT_PARALLEL.value)

    # Test for parallelism: dot(sv, n) == 0
    para_idx, = np.where(np.isclose(np.dot(sv, n), 0, atol=ATOL, rtol=RTOL))

    # dot(s0 - p0, n) is:
    # 0 for coincidence
    # >0 if s0 is same side as n
    # <0 if s0 is other side as n

    prod = np.dot(s0[para_idx] - p0, n)
    idx1 = np.isclose(prod, 0, atol=ATOL, rtol=RTOL)
    idx2 = np.logical_and(~idx1, prod > 0)
    idx3 = np.logical_and(~idx1, prod < 0)

    output[para_idx[idx1]] = ParallelType.PARALLEL_COINCIDENT.value
    output[para_idx[idx2]] = ParallelType.PARALLEL_FRONT.value
    output[para_idx[idx3]] = ParallelType.PARALLEL_BACK.value

    return output


def segment_parallel_to_planes(
    segment: np.ndarray, p0: np.ndarray, n: np.ndarray
) -> np.ndarray:
    """
    Compare a segments to an array of  plane defined by a point and a normal, and return an
    array containing ParallelType information for each segment. PARALLEL_FRONT is returned when
    the segment is on the positive side w.r.t. the normal.
    :param segment: 2x3 segment
    :param p0: Mx3 plane reference points
    :param n: Mx3 plane normals (not necessarily normalized
    :return: length M array of ParallelType
    """

    if not _validate_shape(segment, 2, 3):
        raise ValueError(f"segment has shape {segment.shape} instead of (2, 3)")

    if not _validate_shape(p0, None, 3):
        raise ValueError(f"p0 has shape {p0.shape} instead of (M, 3)")

    if not _validate_shape(n, None, 3):
        raise ValueError(f"n has shape {n.shape} instead of (M, 3)")

    if p0.shape[0] != n.shape[0]:
        raise ValueError(f"p0 and n must have the same first dimension")

    s0 = segment[0]
    s1 = segment[1]
    sv = s1 - s0

    output = np.empty(len(p0), dtype=int)
    output.fill(ParallelType.NOT_PARALLEL.value)

    # Test for parallelism: dot(sv, n) == 0
    para_idx, = np.where(np.isclose(np.dot(n, sv), 0, atol=ATOL, rtol=RTOL))

    # dot(s0 - p0, n) is:
    # 0 for coincidence
    # >0 if s0 is same side as n
    # <0 if s0 is other side as n

    if len(para_idx) > 0:
        prod = np.sum((s0 - p0[para_idx]) * n[para_idx], axis=1)
        idx = np.isclose(prod, 0, atol=ATOL, rtol=RTOL)
        output[para_idx[idx]] = ParallelType.PARALLEL_COINCIDENT.value
        output[para_idx[np.logical_and(~idx, prod > 0)]] = ParallelType.PARALLEL_FRONT.value
        output[para_idx[np.logical_and(~idx, prod < 0)]] = ParallelType.PARALLEL_BACK.value

    return output


def mask_segments(segments: np.array, mask: np.array, diff: bool = True) -> np.array:
    """
    Take a array of 3D segments and masks them with a 2D polygon. The polygon is assumed to be
    perpendicular to the Z axis and the masking is done along the Z axis
    :param segments: Nx2x3 array of segments
    :param mask: Mx2 polygon to be used for masking
    :param diff: if True, the mask area is removed from segment, otherwise the intersection is
    kept
    :return: Lx2x3 array of segments whose length might differ from input
    """

    _validate_segments(segments)

    if len(mask.shape) != 2 or mask.shape[1] != 2:
        raise ValueError(f"mask array must be of dimension (Nx2) instead of {mask.shape}")

    if len(segments) == 0:
        return np.array(segments)

    poly = Polygon(mask)

    # The following is the parallel implementation, which passes the tests but generates
    # a lot of lines (https://github.com/Toblerity/Shapely/issues/779)

    # mls = asMultiLineString(segments)
    # if diff:
    #     mls2 = mls.difference(poly)
    # else:
    #     mls2 = mls.intersection(poly)
    #
    # if mls2.geom_type == "LineString":
    #     return np.array([mls2.coords])
    # elif mls2.geom_type == "MultiLineString":
    #     return np.array([np.array(ls.coords) for ls in mls2])
    # elif mls2.geom_type == "MultiPoint":
    #     return np.empty((0, 2, 3))
    # else:
    #     raise ValueError(f"Unexpected geometry: {mls2.geom_type}")

    # The following is a slower (?), segment-by-segment implementation that does not exhibit
    # the line multiplication issue

    op = "difference" if diff else "intersection"

    output = []
    for i in range(len(segments)):
        ls = asLineString(segments[i, :, :])
        res = getattr(ls, op)(poly)

        # segments with identical start/stop location are sometime returned
        if np.isclose(res.length, 0, atol=ATOL, rtol=RTOL):
            continue

        if res.geom_type == "LineString":
            output.append(np.array([res.coords]))
        elif res.geom_type == "MultiLineString":
            output.append(np.array([np.array(l.coords) for l in res]))

    if output:
        res = np.vstack(output)

        # https://github.com/Toblerity/Shapely/issues/780
        # Some z-coordinate are being erroneously affected by the difference. This ugly
        # hack attempts to restore them.
        for s in segments:
            idx, = np.where(np.all(np.all(s[:, 0:2] == res[:, :, 0:2], axis=2), axis=1))
            res[idx, :, 2] = s[:, 2]

        return res
    else:
        return np.empty((0, 2, 3))


def split_segments(
    segments: np.ndarray, p0: np.ndarray, n: np.ndarray
) -> (np.ndarray, np.ndarray):
    """
    Compute segment to plan intersection and gather them (or their split parts) in two
    categories: behind and in front of the plane along z axis (front: higher z value).
    Segments must not be parallel to the plane and plane must not be parallel to z axis
    Ref: http://geomalgorithms.com/a05-_intersect-1.html
    :param segments: Mx2x3 array of segments
    :param p0: plane reference point (length 3)
    :param n: plane normal (length 3, does not need to be normalized)
    :return: tuple of ?x2x3 array containing segments in front, resp. behind of the plane
    """

    _validate_segments(segments)

    # make sure n points "up"
    if n[2] < 0:
        n = -n

    # if both ends of segments are behind or in front, they can readily be classified as such
    d0 = np.dot(segments[:, 0] - p0, n)
    d1 = np.dot(segments[:, 1] - p0, n)

    behind = np.logical_and(d0 < 0, d1 < 0)
    front = np.logical_and(d0 >= 0, d1 >= 0)
    split_idx, = np.where(np.logical_and(~behind, ~front))

    # Segment parametrized as: s0 + s * sv, where sv = s1 - s0, and s in [0, 1]
    # Plane parametrized as: p0 and n
    # Intersection with plane: s_i = -dot(n, s0-p0) / dot(n, sv)

    split_segs = segments[split_idx, :, :]

    sv = np.diff(split_segs, axis=1).reshape(len(split_segs), 3)
    s0 = split_segs[:, 0]
    s1 = split_segs[:, 1]

    si = -np.dot(s0 - p0, n) / np.dot(sv, n)
    s_mid = s0 + sv * si.reshape((len(s0), 1))
    front_condition = (np.dot(s0 - p0, n) >= 0).reshape((len(s0), 1))

    split_segs_front = np.empty_like(split_segs)
    split_segs_front[:, 0, :] = np.where(front_condition, s0, s_mid)
    split_segs_front[:, 1, :] = np.where(front_condition, s_mid, s1)

    split_segs_back = np.empty_like(split_segs)
    split_segs_back[:, 0, :] = np.where(front_condition, s_mid, s0)
    split_segs_back[:, 1, :] = np.where(front_condition, s1, s_mid)

    # gather all segments together
    front_segments = np.vstack([segments[front], split_segs_front])
    back_segments = np.vstack([segments[behind], split_segs_back])
    return front_segments, back_segments


DEGENERATE = -1  # ignore face
NO_INT_PARALLEL_FRONT = 0  # add face to mask if segment is rear
NO_INT_PARALLEL_BEHIND = 1  # add face to mask if segment is rear
NO_INT_SHORT_SEGMENT_FRONT = 2  # add face to mask if r > 0 (seg behind)
NO_INT_SHORT_SEGMENT_BEHIND = 3  # add face to mask if r > 0 (seg behind)
NO_INT_OUTSIDE_FACE = 4  # add face to mask if rear half-seg 2D intersects
INT_COPLANAR = 5  # ignore face
INT_INSIDE = 6  # 3 sub-face
INT_EDGE = 7  # 2 sub-face
INT_VERTEX = 8  # add face to mask if rear half-seg 2D intersects


def segment_triangle_intersection(segment, triangle):
    """
    Unoptimized version of single triangle intersection computation based on
    http://geomalgorithms.com/a06-_intersect-2.html#intersect3D_RayTriangle()

    :param segment: 2x3
    :param triangle: 3x3
    :return: -1 degenerate triangle
    """
    # //    Input:  a ray R, and a triangle T
    # //    Output: intersection = intersection point (when it exists)
    # //    Return: -1 = triangle is degenerate (a segment or point)
    # //             0 =  disjoint (no intersect)
    # //             1 =  intersect in unique point I1
    # //             2 =  are in the same plane

    if not _validate_shape(segment, 2, 3):
        raise ValueError(f"segment has shape {segment.shape} instead of (2, 3)")

    if not _validate_shape(triangle, 3, 3):
        raise ValueError(f"faces has shape {triangle.shape} instead of (M, 3, 3)")

    #     // get triangle edge vectors and plane normal
    #     u = T.V1 - T.V0;
    u = triangle[1] - triangle[0]
    #     v = T.V2 - T.V0;

    v = triangle[2] - triangle[0]
    #     n = u * v;              // cross product
    n = np.cross(u, v)

    # make sure n points upwards
    if n[2] < 0:
        n = -n
        u, v = v, u
        swap_st = True
    else:
        swap_st = False

    #     sv = R.P1 - R.P0;              // ray direction vector
    sv = segment[1] - segment[0]
    s0 = segment[0]

    #     if (n == (Vector)0)             // triangle is degenerate
    #         return -1;                  // do not deal with this case
    if np.isclose(np.linalg.norm(n), 0, atol=ATOL):
        return DEGENERATE, 0, 0, 0, (0, 0, 0), 0

    #     w0 = R.P0 - T.V0;
    w0 = s0 - triangle[0]

    #     a = -dot(n,w0);
    a = -np.dot(n, w0)

    #     b = dot(n,sv);
    b = np.dot(n, sv)

    #     if (fabs(b) < SMALL_NUM) {     // ray is  parallel to triangle plane
    #         if (a == 0)                 // ray lies in triangle plane
    #             return 2;
    #         else return 0;              // ray disjoint from plane
    #     }
    atol = np.linalg.norm(sv) * RTOL
    if np.isclose(b, 0, atol=atol):
        if np.isclose(a, 0, atol=atol):
            return INT_COPLANAR, None, None, None, None, 0
        elif a <= 0:
            return NO_INT_PARALLEL_FRONT, None, None, None, None, 0
        else:
            return NO_INT_PARALLEL_BEHIND, None, None, None, None, 0

    #     // get intersect point of ray with triangle plane
    #     r = a / b;
    r = a / b

    if np.isclose(r, 0, atol=ATOL):
        r = 0
    elif np.isclose(r, 1, atol=ATOL):
        r = 1

    #     if (r < 0.0)                    // ray goes away from triangle
    #         return 0;                   // => no intersect
    if r < 0 or r > 1:
        if a <= 0:
            return NO_INT_SHORT_SEGMENT_FRONT, r, None, None, None, b
        else:
            return NO_INT_SHORT_SEGMENT_BEHIND, r, None, None, None, b

    # intersection = R.P0 + r * sv;            // intersect point of ray and plane
    intersection = s0 + r * sv

    #     // is I inside T?
    #     float    uu, uv, vv, wu, wv, D;
    #     uu = dot(u,u);
    uu = np.dot(u, u)

    #     uv = dot(u,v);
    uv = np.dot(u, v)

    #     vv = dot(v,v);
    vv = np.dot(v, v)

    #     w = intersection - T.V0;
    w = intersection - triangle[0]

    #     wu = dot(w,u);
    wu = np.dot(w, u)

    #     wv = dot(w,v);
    wv = np.dot(w, v)

    #     D = uv * uv - uu * vv;
    d = uv * uv - uu * vv

    #     // get and test parametric coords
    #     float s, t;
    #     s = (uv * wv - vv * wu) / D;
    s = (uv * wv - vv * wu) / d
    if np.isclose(s, 0, atol=ATOL):
        s = 0
    elif np.isclose(s, 1, atol=ATOL):
        s = 1

    #     if (s < 0.0 || s > 1.0)         // I is outside T
    #         return 0;
    if s < 0 or s > 1:
        if swap_st:
            return NO_INT_OUTSIDE_FACE, r, None, s, intersection, b
        else:
            return NO_INT_OUTSIDE_FACE, r, s, None, intersection, b

    #     t = (uv * wu - uu * wv) / D;
    t = (uv * wu - uu * wv) / d

    if np.isclose(t, 0, atol=ATOL):
        t = 0
    elif np.isclose(t, 1, atol=ATOL):
        t = 1

    st = s + t
    if np.isclose(st, 1, atol=ATOL):
        st = 1
        t = 1 - s  # ensure consistency

    #     if (t < 0.0 || (s + t) > 1.0)  // I is outside T
    #         return 0;
    if t < 0 or st > 1.0:
        if swap_st:
            return NO_INT_OUTSIDE_FACE, r, t, s, intersection, b
        else:
            return NO_INT_OUTSIDE_FACE, r, s, t, intersection, b

    new_intersection = triangle[0] + s * u + t * v

    if (s == 0 and t == 0) or (s == 1 and t == 0) or (s == 0 and t == 1):
        if swap_st:
            return INT_VERTEX, r, t, s, new_intersection, b
        else:
            return INT_VERTEX, r, s, t, new_intersection, b

    if s == 0 or t == 0 or st == 1:
        if swap_st:
            return INT_EDGE, r, t, s, new_intersection, b
        else:
            return INT_EDGE, r, s, t, new_intersection, b

    #     return 1;                       // I is in T
    if swap_st:
        return INT_INSIDE, r, t, s, intersection, b
    else:
        return INT_INSIDE, r, s, t, intersection, b


def segment_triangles_intersection(segment, triangles, callback):
    """
    Vectorized version of single triangle intersection computation based on
    http://geomalgorithms.com/a06-_intersect-2.html#intersect3D_RayTriangle()

    TODO: to be completed

    :param segment: 2x3
    :param triangles: Mx3x3
    :param callback: callback
    :return: to be completed
    """

    if not _validate_shape(segment, 2, 3):
        raise ValueError(f"segment has shape {segment.shape} instead of (2, 3)")

    if not _validate_shape(triangles, None, 3, 3):
        raise ValueError(f"faces has shape {triangles.shape} instead of (M, 3, 3)")

    #     // get triangle edge vectors and plane normal
    u = triangles[:, 1] - triangles[:, 0]
    v = triangles[:, 2] - triangles[:, 0]
    n = np.cross(u, v)

    # =============== #
    # DEGENERATE CASE #
    # =============== #

    # compute & test stuff
    degenerate_idx = np.isclose(np.linalg.norm(n, axis=1), 0, atol=ATOL)

    # return results
    if np.any(degenerate_idx):
        callback(segment, triangles[degenerate_idx], DEGENERATE, None, None, None, None, None)

    # compute remaining data to process
    rest_idx = ~degenerate_idx
    if ~np.any(rest_idx):
        return

    # ============= #
    # PARALLEL CASE #
    # ============= #

    triangles = triangles[rest_idx, ...]
    n = n[rest_idx, ...]
    u = u[rest_idx, ...]
    v = v[rest_idx, ...]

    # make sure n points upwards
    swap_idx = n[:, 2] < 0
    if np.any(swap_idx):
        n[swap_idx] = -n[swap_idx]
        u[swap_idx], v[swap_idx] = v[swap_idx], u[swap_idx]

    # compute
    sv = segment[1] - segment[0]
    s0 = segment[0]
    atol = np.linalg.norm(sv) * RTOL
    w0 = s0 - triangles[:, 0]
    a = -np.sum(n * w0, axis=1)
    b = np.dot(n, sv)

    # test
    parallel_idx = np.isclose(b, 0, atol=atol)
    coplanar_idx = parallel_idx & np.isclose(a, 0, atol=atol)
    para_front = parallel_idx & ~coplanar_idx & (a <= 0)
    para_behind = parallel_idx & ~coplanar_idx & ~para_front

    # return results
    for idx, code in [
        (coplanar_idx, INT_COPLANAR),
        (para_front, NO_INT_PARALLEL_FRONT),
        (para_behind, NO_INT_PARALLEL_BEHIND),
    ]:
        if np.any(idx):
            callback(segment, triangles[idx], code, None, None, None, None, None)

    # compute remaining data to process
    rest_idx = ~parallel_idx
    if ~np.any(rest_idx):
        return

    # ================== #
    # SHORT SEGMENT CASE #
    # ================== #

    triangles = triangles[rest_idx]
    swap_idx = swap_idx[rest_idx]
    u = u[rest_idx]
    v = v[rest_idx]
    a = a[rest_idx]
    b = b[rest_idx]

    # compute
    r = a / b
    r[np.isclose(r, 0, atol=ATOL)] = 0
    r[np.isclose(r, 1, atol=ATOL)] = 1

    # test
    short_idx = (r < 0) | (r > 1)
    short_front_idx = short_idx & (a <= 0)
    short_behind_idx = short_idx & ~short_front_idx

    # return results
    for idx, code in [
        (short_front_idx, NO_INT_SHORT_SEGMENT_FRONT),
        (short_behind_idx, NO_INT_SHORT_SEGMENT_BEHIND),
    ]:
        if np.any(idx):
            callback(segment, triangles[idx], code, r[idx], None, None, None, None)

    # compute remaining data to process
    rest_idx = ~short_idx
    if ~np.any(rest_idx):
        return

    # ==================================== #
    # SINGLE-POINT PLANE INTERSECTION CASE #
    # ==================================== #

    triangles = triangles[rest_idx]
    swap_idx = swap_idx[rest_idx]
    b = b[rest_idx]
    u = u[rest_idx]
    v = v[rest_idx]
    r = r[rest_idx]

    # compute
    itrsct = (
        np.ones(shape=(len(r), 1)) @ s0[np.newaxis, :] + r[:, np.newaxis] @ sv[np.newaxis, :]
    )
    uu = np.sum(u * u, axis=1)
    uv = np.sum(u * v, axis=1)
    vv = np.sum(v * v, axis=1)
    w = itrsct - triangles[:, 0]
    wu = np.sum(w * u, axis=1)
    wv = np.sum(w * v, axis=1)
    d = uv * uv - uu * vv
    s = (uv * wv - vv * wu) / d
    t = (uv * wu - uu * wv) / d
    st = s + t

    # snap important value to 0 or 1
    ss0 = np.isclose(s, 0, atol=ATOL)
    ss1 = np.isclose(s, 1, atol=ATOL)
    tt0 = np.isclose(t, 0, atol=ATOL)
    tt1 = np.isclose(t, 1, atol=ATOL)
    st1 = np.isclose(st, 1, atol=ATOL)
    s[ss0] = 0
    s[ss1] = 1
    t[tt0] = 0
    t[tt1] = 1
    st[st1] = 1
    t[st1] = 1 - s[st1]  # this is for consistency

    # test
    outside_idx = (s < 0) | (s > 1) | (t < 0) | (st > 1)
    vertex_idx = ~outside_idx & ((ss0 & tt0) | (ss0 & tt1) | (ss1 & tt0))
    edge_idx = ~outside_idx & ~vertex_idx & (ss0 | tt0 | st1)
    inside_idx = ~outside_idx & ~vertex_idx & ~edge_idx

    # return results, swapping s and t as required
    s[swap_idx], t[swap_idx] = t[swap_idx], s[swap_idx]
    for idx, code in [
        (outside_idx, NO_INT_OUTSIDE_FACE),
        (vertex_idx, INT_VERTEX),
        (edge_idx, INT_EDGE),
        (inside_idx, INT_INSIDE),
    ]:
        if np.any(idx):
            callback(
                segment, triangles[idx], code, r[idx], s[idx], t[idx], b[idx], itrsct[idx]
            )


# noinspection DuplicatedCode
def mask_segment(segment: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Compute the visibility of the segment given the provided faces.
    Segments are hidden when the are on the opposite side of the normal
    :param segment: (2x3) the segment to process
    :param faces: (Nx3x3) the faces to consider
    :return: (Mx2x3) list of visible sub-segments
    """

    if not _validate_shape(segment, 2, 3):
        raise ValueError(f"segment has shape {segment.shape} instead of (2, 3)")

    if not _validate_shape(faces, None, 3, 3):
        raise ValueError(f"faces has shape {faces.shape} instead of (M, 3, 3)")

    # Check 2D overlap, all non-overlapping faces can be ignored
    active = triangles_overlap_segment_2d(faces, segment)
    if not np.any(active):
        return segment.reshape((1, 2, 3))

    # Iterate over every triangle and run the intersection function
    mask = []
    for i, face in enumerate(faces[active]):
        res, r, s, t, itrsct, b = segment_triangle_intersection(segment, face)

        if res in (
            DEGENERATE,
            NO_INT_SHORT_SEGMENT_FRONT,
            NO_INT_PARALLEL_FRONT,
            INT_COPLANAR,
        ):
            # this face is not masking the segment
            continue
        elif res in (NO_INT_PARALLEL_BEHIND, NO_INT_SHORT_SEGMENT_BEHIND):
            # this face is masking the segment
            mask.append(face)
        elif res == NO_INT_OUTSIDE_FACE:
            # segment crosses the face's plane, but not the face itself
            # masking occurs if the rear half-segment 2D-overlaps the face
            if b > 0:
                test_seg = np.array([segment[0], itrsct])
            else:
                test_seg = np.array([itrsct, segment[1]])

            overlap = triangles_overlap_segment_2d(
                np.array([face]), np.array(test_seg), accept_vertex_only=False
            )
            if overlap[0]:
                mask.append(face)
        elif res == INT_INSIDE:
            # lets consider the 3 sub-faces with itrsct as vertex, at least one should 2D-
            # intersect with the rear half-segment and should be added to the mask

            if b > 0:
                test_seg = np.array([segment[0], itrsct])
            elif b < 0:
                test_seg = np.array([itrsct, segment[1]])
            else:
                continue

            subfaces = np.array(
                [
                    (face[0], face[1], itrsct),
                    (face[1], face[2], itrsct),
                    (face[2], face[0], itrsct),
                ]
            )
            overlap = triangles_overlap_segment_2d(
                subfaces, test_seg, accept_vertex_only=False
            )
            if np.any(overlap):
                mask.append(subfaces[np.argmax(overlap)])
            else:
                logging.warning(
                    f"inconsistent INTERSECTION_INSIDE with segment {segment} and "
                    f"face {face}: no overlapping sub-face"
                )
        elif res == INT_EDGE:
            # in this case, itrsct defines two sub-faces, at least one should 2D-intersect with
            # the rear half-segment and should be added to the mask
            if b > 0:
                test_seg = np.array([segment[0], itrsct])
            elif b < 0:
                test_seg = np.array([itrsct, segment[1]])
            else:
                continue

            if ~np.isclose(np.linalg.norm(test_seg[1] - test_seg[0]), 0, atol=ATOL, rtol=RTOL):
                if s == 0:
                    subfaces = np.array(
                        [(face[0], face[1], itrsct), (itrsct, face[1], face[2])]
                    )
                elif t == 0:
                    subfaces = np.array(
                        [(face[0], itrsct, face[2]), (itrsct, face[1], face[2])]
                    )
                else:
                    subfaces = np.array(
                        [(face[0], itrsct, face[2]), (face[0], face[1], itrsct)]
                    )
                overlap = triangles_overlap_segment_2d(
                    subfaces, test_seg, accept_vertex_only=False
                )

                mask.extend(subfaces[overlap])
        elif res == INT_VERTEX:
            # in that case we add the face to the mask if the rear half-segment 2D intersects
            # with the face
            if b > 0:
                test_seg = np.array([segment[0], itrsct])
            elif b < 0:
                test_seg = np.array([itrsct, segment[1]])
            else:
                continue

            if ~np.isclose(np.linalg.norm(test_seg[1] - test_seg[0]), 0, atol=ATOL, rtol=RTOL):
                overlap = triangles_overlap_segment_2d(
                    np.array([face]), test_seg, accept_vertex_only=False
                )
                if overlap[0]:
                    mask.append(face)
        else:
            logging.warning(
                f"inconsistent result code {res} from segment_triangle_intersection with "
                f"segment {segment} and face {face}"
            )

    # apply mask on segment
    polys = []
    for f in mask:
        p = Polygon(f[:, 0:2].round(decimals=14))
        if p.is_valid:
            polys.append(p)
    msk_seg = asLineString(segment).difference(shapely.ops.unary_union(polys))
    # TODO: cases where we might want to keep a point
    # - seg parallel to camera axis
    # - others?
    if np.isclose(msk_seg.length, 0, atol=ATOL, rtol=RTOL):
        # segments with identical start/stop location are sometime returned
        return np.empty(shape=(0, 2, 3))
    elif msk_seg.geom_type == "LineString":
        return np.array([msk_seg.coords])
    elif msk_seg.geom_type == "MultiLineString":
        output = np.array([np.array(l.coords) for l in msk_seg if l.length > 1e-14])
        return np.empty(shape=(0, 2, 3)) if len(output) == 0 else output


# noinspection DuplicatedCode
def mask_segment_parallel(segment: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Compute the visibility of the segment given the provided faces.
    Segments are hidden when the are on the opposite side of the normal
    :param segment: (2x3) the segment to process
    :param faces: (Nx3x3) the faces to consider
    :return: (Mx2x3) list of visible sub-segments
    """

    if not _validate_shape(segment, 2, 3):
        raise ValueError(f"segment has shape {segment.shape} instead of (2, 3)")

    if not _validate_shape(faces, None, 3, 3):
        raise ValueError(f"faces has shape {faces.shape} instead of (M, 3, 3)")

    # Check 2D overlap, all non-overlapping faces can be ignored
    active = triangles_overlap_segment_2d(faces, segment)
    if not np.any(active):
        return segment.reshape((1, 2, 3))

    mask = []

    # noinspection PyShadowingNames,PyUnusedLocal
    def callback(segment, triangles, res, r, s, t, b, itrsct):
        nonlocal mask

        if res in (NO_INT_PARALLEL_BEHIND, NO_INT_SHORT_SEGMENT_BEHIND):
            # this face is masking the segment
            mask.extend(triangles)
            return
        elif res in [
            DEGENERATE,
            NO_INT_SHORT_SEGMENT_FRONT,
            NO_INT_PARALLEL_FRONT,
            INT_COPLANAR,
        ]:
            return

        # build an array of rear half-segments
        half_segments = np.repeat(segment.reshape(1, 2, 3), len(triangles), axis=0)
        idx = b > 0
        half_segments[idx, 1, :] = itrsct[idx, :]
        half_segments[~idx, 0, :] = itrsct[~idx, :]

        if res == NO_INT_OUTSIDE_FACE:
            # segment crosses the face's plane, but not the face itself
            # masking occurs if the rear half-segment 2D-overlaps the face
            overlap = triangles_overlap_segment_2d(
                triangles, half_segments, accept_vertex_only=False
            )
            mask.extend(triangles[overlap])
        elif res == INT_INSIDE:
            # lets consider the 3 sub-faces with itrsct as vertex, at least one should 2D-
            # intersect with the rear half-segment and should be added to the mask

            # TODO: this case is not vectorized but is rather rare anyway

            for i in range(len(triangles)):
                subfaces = np.array(
                    [
                        (triangles[i, 0], triangles[i, 1], itrsct[i]),
                        (triangles[i, 1], triangles[i, 2], itrsct[i]),
                        (triangles[i, 2], triangles[i, 0], itrsct[i]),
                    ]
                )
                overlap = triangles_overlap_segment_2d(
                    subfaces, half_segments[i], accept_vertex_only=False
                )
                if np.any(overlap):
                    mask.append(subfaces[np.argmax(overlap)])
                else:
                    logging.warning(
                        f"inconsistent INTERSECTION_INSIDE with segment {segment} and "
                        f"face {triangles[i]}: no overlapping sub-face"
                    )
        elif res == INT_EDGE:
            # in this case, itrsct defines two sub-faces, at least one should 2D-intersect with
            # the rear half-segment and should be added to the mask

            idx, = np.nonzero(
                ~np.isclose(
                    np.linalg.norm(half_segments[:, 1] - half_segments[:, 0], axis=1),
                    0,
                    atol=ATOL,
                    rtol=RTOL,
                )
            )

            for i in idx:
                if s[i] == 0:
                    subfaces = np.array(
                        [
                            (triangles[i, 0], triangles[i, 1], itrsct[i]),
                            (itrsct[i], triangles[i, 1], triangles[i, 2]),
                        ]
                    )
                elif t[i] == 0:
                    subfaces = np.array(
                        [
                            (triangles[i, 0], itrsct[i], triangles[i, 2]),
                            (itrsct[i], triangles[i, 1], triangles[i, 2]),
                        ]
                    )
                else:
                    subfaces = np.array(
                        [
                            (triangles[i, 0], itrsct[i], triangles[i, 2]),
                            (triangles[i, 0], triangles[i, 1], itrsct[i]),
                        ]
                    )
                overlap = triangles_overlap_segment_2d(
                    subfaces, half_segments[i], accept_vertex_only=False
                )

                mask.extend(subfaces[overlap])
        elif res == INT_VERTEX:
            # in that case we add the face to the mask if the rear half-segment 2D intersects
            # with the face

            idx = ~np.isclose(
                np.linalg.norm(half_segments[:, 1] - half_segments[:, 0], axis=1),
                0,
                atol=ATOL,
                rtol=RTOL,
            )

            overlap = triangles_overlap_segment_2d(
                triangles[idx], half_segments[idx], accept_vertex_only=False
            )
            mask.extend(triangles[idx][overlap])

    segment_triangles_intersection(segment, faces[active], callback)

    # apply mask on segment
    polys = []
    for f in mask:
        p = Polygon(f[:, 0:2].round(decimals=14))
        if p.is_valid:
            polys.append(p)
    msk_seg = asLineString(segment).difference(shapely.ops.unary_union(polys))

    geom_type = msk_seg.geom_type
    length = msk_seg.length

    # TODO: cases where we might want to keep a point
    # - seg parallel to camera axis
    # - others?
    if np.isclose(length, 0, atol=ATOL, rtol=RTOL):
        # segments with identical start/stop location are sometime returned
        output = np.empty(shape=(0, 2, 3))
    elif geom_type == "LineString":
        output = np.array([msk_seg.coords])
    elif geom_type == "MultiLineString":
        output = np.array([np.array(l.coords) for l in msk_seg if l.length > 1e-14])
        if len(output) == 0:
            output.shape = (0, 2, 3)
    else:
        output = np.empty(shape=(0, 2, 3))

    return output
