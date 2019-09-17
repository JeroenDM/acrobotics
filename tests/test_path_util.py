import numpy as np
from numpy.testing import assert_almost_equal

from acrobotics.util import pose_x, pose_y, pose_z
from acrobotics.path.util import (
    xyz_intrinsic_to_rot_mat,
    rotation_matrix_to_rxyz,
    check_rxyz_input,
    tf_inverse,
    is_in_range,
)


def test_rot_to_rpy():
    for _ in range(100):
        rpy = np.random.rand(3) * np.pi - np.pi / 2
        rpy[0] = rpy[0] * 2
        rpy[2] = rpy[2] * 2

        if not check_rxyz_input(rpy):
            continue

        R = xyz_intrinsic_to_rot_mat(rpy)
        new_rpy = rotation_matrix_to_rxyz(R)
        assert_almost_equal(rpy, new_rpy)


def test_tf_inverse():
    for _ in range(30):
        for pose in [pose_x, pose_y, pose_z]:
            v = np.random.rand(4)
            tf = pose(*v)
            identity = np.eye(4)
            assert_almost_equal(tf_inverse(tf) @ tf, identity)
            assert_almost_equal(tf @ tf_inverse(tf), identity)


def test_is_in_range():
    assert is_in_range(2, -2, 3)
    assert not is_in_range(2, -2, 1)
    assert is_in_range(-5, -6, -3)
    assert not is_in_range(-5, -4, -3)
