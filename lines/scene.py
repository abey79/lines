import math
from typing import Sequence

import numpy as np
import tqdm

from .math import (
    segments_parallel_to_face,
    ParallelType,
    mask_segments,
    split_segments,
    segments_outside_triangle_2d,
    mask_segment_parallel)
from .rendered_scene import RenderedScene
from .shapes import Node


class Scene(Node):
    def __init__(self):
        super().__init__()
        self._camera_matrix = np.identity(4)

    def look_at(self, eye: Sequence[float], to: Sequence[float]) -> None:
        eye = np.array(eye, dtype="float64")
        to = np.array(to, dtype="float64")
        up = np.array([0, 0, 1], dtype="float64")

        f = to - eye
        f = f / np.linalg.norm(f)
        s = np.cross(f, up)
        s = s / np.linalg.norm(s)
        u = np.cross(s, f)
        u = u / np.linalg.norm(u)

        m = np.zeros((4, 4))
        m[0, :-1] = s
        m[1, :-1] = u
        m[2, :-1] = -f
        m[3, :-1] = eye
        m[-1, -1] = 1.0

        m = np.transpose(np.linalg.inv(m))

        self._camera_matrix = m

    def frustum(self, l: float, r: float, b: float, t: float, n: float, f: float) -> None:
        t1 = 2 * n
        t2 = r - l
        t3 = t - b
        t4 = f - n
        # Note: the 3rd line of the matrix has opposite sign w.r.t the ln project, in order
        # to maintain positive Z values for geometries closer to the camera
        frustum_matrix = np.array(
            [
                [t1 / t2, 0, (r + l) / t2, 0],
                [0, t1 / t3, (t + b) / t3, 0],
                [0, 0, (f + n) / t4, (t1 * f) / t4],
                [0, 0, -1, 0],
            ]
        )
        self._camera_matrix = frustum_matrix @ self._camera_matrix

    def perspective(self, fov: float, near: float, far: float) -> None:
        y_max = near * math.tan(fov * math.pi / 180 / 2)
        x_max = y_max * 1  # y_max * aspect
        self.frustum(-x_max, x_max, -y_max, y_max, near, far)

    def render(self, renderer: str = "v2", merge_lines: bool = True) -> RenderedScene:
        """
        Renders the scene as it is currently setup to a lines.RenderedScene object.
        :param renderer: "v1" or "v2"
        :param merge_lines: if True, lines.RenderedScene.optimize() will be called after
        the rendering process is completed
        :return:
        """
        segments, faces = self.compile(self._camera_matrix)
        if renderer == "v1":
            vectors = self._render_v1(segments, faces)
        elif renderer == "v2":
            vectors = self._render_v2(segments, faces)
        else:
            raise ValueError(f"renderer type '{renderer}' unsupported, us 'v1' or 'v2'")

        return RenderedScene(self, vectors, merge_lines, segments, faces)

    @staticmethod
    def _render_v1(all_segments, all_faces) -> np.ndarray:
        """
        FIXME
        Renders the scene with the current camera projection. Returns a
        shapely.geometry.MultiLineString object containing all the visible 2D segment
        :return: the rendered 2D lines
        """

        # (B) Crop anything that is not in the frustum
        # TODO: should also crop in the Z-direction

        all_segments = mask_segments(
            all_segments, np.array(((-1, -1), (-1, 1), (1, 1), (1, -1))), False
        )

        # (C) Process all face/segment occlusion as follows:
        #
        # (1) Face parallel to camera ray (n.z == 0)
        #     -> all segments -> unmasked
        #
        # (2) If the 2D projected segment does not intersect the face
        #     -> unmasked
        #
        # (3) Segment and face are parallel
        #     -> segment contained in face plan -> unmasked
        #     -> segment in front of face plan -> unmasked
        #     -> segment behind the face plan -> to be masked
        #
        # (4) Segment is not parallel, split it at plan intersection
        #     -> half segment in front -> unmasked
        #     -> half segment behind -> to be masked
        #
        # (5) All (sub-)segment flagged as to be masked are... masked with face.

        face_normals = np.cross(
            all_faces[:, 1] - all_faces[:, 0], all_faces[:, 2] - all_faces[:, 0]
        )
        non_perp_idx = face_normals[:, 2] != 0

        for (p0, p1, p2), n in zip(all_faces[non_perp_idx], face_normals[non_perp_idx]):
            # All segments strictly outside of the face (in 2D) can be left alone
            outside = segments_outside_triangle_2d(all_segments, np.array([p0, p1, p2]))
            segments_to_process = all_segments[~outside]

            # Check parallelism
            para = segments_parallel_to_face(segments_to_process, p0, n)

            idx_masked, = np.where(para == ParallelType.PARALLEL_BACK.value)
            idx_unmasked, = np.where(
                np.logical_or(
                    para == ParallelType.PARALLEL_FRONT.value,
                    para == ParallelType.PARALLEL_COINCIDENT.value,
                )
            )
            idx_split, = np.where(para == ParallelType.NOT_PARALLEL.value)

            # Split the required segments in halves.
            segs_front, segs_back = split_segments(segments_to_process[idx_split], p0, n)

            # Mask everything that needs to be
            masked_segments = mask_segments(
                np.vstack([segments_to_process[idx_masked], segs_back]),
                np.array([p0[0:2], p1[0:2], p2[0:2]]),
                True,
            )

            # Collect all segments for the next iteration
            all_segments = np.vstack(
                [
                    all_segments[outside],
                    segments_to_process[idx_unmasked],
                    segs_front,
                    masked_segments,
                ]
            )

        # (D) Convert to 2D data
        return all_segments[:, :, 0:2]

    @staticmethod
    def _render_v2(all_segments: np.ndarray, all_faces: np.ndarray) -> np.ndarray:
        # Crop anything that is not in the frustum
        # TODO: should also crop in the Z-direction

        all_segments = mask_segments(
            all_segments, np.array(((-1, -1), (-1, 1), (1, 1), (1, -1))), False
        )

        if len(all_segments) == 0:
            return np.empty(shape=(0, 2, 2))

        # For each segment, hide some or all of it based on faces that may be in front.
        results = []
        # for i, s in tqdm.tqdm(enumerate(all_segments), total=len(all_segments)):
        for s in tqdm.tqdm(all_segments):
            results.append(mask_segment_parallel(s, all_faces))
            _ = np.vstack(results)
        output = np.vstack(results)

        # Convert to 2D data
        return output[:, :, 0:2]
