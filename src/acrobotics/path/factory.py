import numpy as np

from copy import deepcopy
from typing import List
from numpy.linalg import norm

from acrolib.quaternion import Quaternion

from .path_pt import TolPositionPt, PathPt


def check_num_points(num_points: int):
    if num_points < 2:
        raise Exception(f"Value of num_points must be 2 or more, not {num_points}.")


def create_line(start_pt: PathPt, end_pos: np.ndarray, num_points: int) -> List[PathPt]:
    """ Copy a given toleranced PathPt along a straight line."""
    check_num_points(num_points)
    trans_vec = (end_pos - start_pt.pos) / (num_points - 1)
    path = [start_pt]
    for _ in range(num_points - 1):
        new_pt = deepcopy(path[-1])
        new_pt.translate(trans_vec)
        path.append(new_pt)
    return path


def create_circle(
    start_pt: PathPt, mid_point: np.ndarray, rotation_axis: np.ndarray, num_points: int
):
    """Copy a given toleranced PathPt along a circle with a given mid point and rotation axis."""
    check_num_points(num_points)
    return create_arc(start_pt, mid_point, rotation_axis, 2 * np.pi, num_points)


def create_arc(
    start_pt: PathPt, mid_point: np.ndarray, rotation_axis, angle, num_points
):
    """Copy a given toleranced PathPt along an arc with a given mid point and rotation axis."""
    check_num_points(num_points)
    rotation_axis = rotation_axis / norm(rotation_axis)

    rotating_vector = start_pt.pos - mid_point
    a = np.linspace(angle / (num_points - 1), angle, num_points - 1)

    path = [deepcopy(start_pt)]
    for ai in a:
        rot_mat = Quaternion(angle=ai, axis=rotation_axis).rotation_matrix
        offset = (rot_mat @ rotating_vector) - rotating_vector
        new_pt = deepcopy(start_pt)
        new_pt.translate(offset)
        new_pt.rotate(rot_mat)
        path.append(new_pt)

    return path
