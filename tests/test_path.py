import pytest
import numpy as np
from numpy.testing import assert_almost_equal

from acrolib.quaternion import Quaternion
from acrolib.sampling import SampleMethod
from acrolib.geometry import rotation_matrix_to_rpy

from acrobotics.robot import Robot, IKResult

from acrobotics.path.tolerance import (
    Tolerance,
    NoTolerance,
    SymmetricTolerance,
    QuaternionTolerance,
)

from acrobotics.path.path_pt_base import PathPt
from acrobotics.path.path_pt import TolEulerPt, TolPositionPt, TolQuatPt
from acrobotics.path.sampling import SamplingSetting, SearchStrategy

IK_RESULT = IKResult(True, [np.ones(6), 2 * np.ones(6)])


class DummyRobot(Robot):
    def __init__(self, is_colliding=False):
        self.is_colliding = is_colliding
        self.ndof = 6

    def is_in_collision(self, joint_position, scene=None):
        return self.is_colliding

    def ik(self, transform):
        return IK_RESULT


def assert_in_range(x, lower, upper):
    assert x <= upper
    assert x >= lower


class TestPathPt:
    def test_calc_ik(self):
        samples = [[], []]
        joint_solutions = PathPt._calc_ik(DummyRobot(), samples)
        assert len(joint_solutions) == 4

        assert_almost_equal(joint_solutions[0], IK_RESULT.solutions[0])
        assert_almost_equal(joint_solutions[1], IK_RESULT.solutions[1])
        assert_almost_equal(joint_solutions[2], IK_RESULT.solutions[0])
        assert_almost_equal(joint_solutions[3], IK_RESULT.solutions[1])


class TestTolPositionPt:
    def test_create(self):
        TolPositionPt([1, 2, 3], Quaternion(), 3 * [NoTolerance()])

    def test_sample_grid(self):
        q = Quaternion()
        pos_tol = [Tolerance(-0.5, 0.5, 2), Tolerance(-0.5, 0.5, 2), NoTolerance()]

        point = TolPositionPt([0.5, 0.5, 1], q, pos_tol)
        grid = point.sample_grid()
        position_samples = [[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
        for pos, T in zip(position_samples, grid):
            assert_almost_equal(pos, T[:3, 3])

    def test_sample_grid_2(self):
        """
        Rotate with pi / 2 around z, give a tolerance along x, 
        expect a grid sampled along y
        """
        tol = [SymmetricTolerance(0.1, 3), NoTolerance(), NoTolerance()]
        pt = TolPositionPt(
            [1, 2, 3], Quaternion(angle=np.pi / 2, axis=np.array([0, 0, 1])), tol
        )

        tf_samples = pt.sample_grid()
        g = [tf[:3, 3] for tf in tf_samples]
        g_desired = np.array([[1, 1.9, 3], [1, 2, 3], [1, 2.1, 3]])
        assert_almost_equal(g, g_desired)

    def test_sample_incremental_1(self):
        """
        Rotate with pi / 2 around z, give a tolerance along x, 
        expect a grid sampled along y
        """
        tol = [SymmetricTolerance(0.1, 3), NoTolerance(), NoTolerance()]
        pt = TolPositionPt(
            [1, 2, 3], Quaternion(angle=np.pi / 2, axis=np.array([0, 0, 1])), tol
        )

        g = pt.sample_incremental(10, SampleMethod.random_uniform)
        pos_samples = [tf[:3, 3] for tf in g]

        for sample in pos_samples:
            assert_almost_equal(sample[0], 1)
            assert_almost_equal(sample[2], 3)

            assert_in_range(sample[1], 1.9, 2.1)

    def test_sample_incremental_2(self):
        tol = [SymmetricTolerance(0.1, 3), Tolerance(0.2, 1.0, 3), NoTolerance()]
        pt = TolPositionPt(
            [1, 2, 3], Quaternion(angle=np.pi / 2, axis=np.array([0, 0, 1])), tol
        )

        g = pt.sample_incremental(10, SampleMethod.random_uniform)

        pos_samples = [tf[:3, 3] for tf in g]
        for sample in pos_samples:
            assert_almost_equal(sample[2], 3)

            assert_in_range(sample[0], 0, 3)  # 1 + (-1, 2)
            assert_in_range(sample[1], 1.9, 2.1)  # 2 + (-0.1, 0.1)

    def test_transform_to_rel_tolerance_deviation(self):
        tol = [SymmetricTolerance(0.1, 3), Tolerance(0.2, 1.0, 3), NoTolerance()]
        quat = Quaternion(angle=np.pi / 2, axis=np.array([0, 0, 1]))
        pt = TolPositionPt([1, 2, 3], quat, tol)

        tf = quat.transformation_matrix
        tf[:3, 3] = np.array([1.06, 1.91, 3])

        p_rel = pt.transform_to_rel_tolerance_deviation(tf)
        p_desired = np.array([-0.09, -0.06, 0.0])

        assert_almost_equal(p_rel, p_desired)


class TestEulerPt:
    def test_grid(self):
        pos = [1, 2, 3]
        pos_tol = [NoTolerance()] * 3
        rot_tol = [NoTolerance(), NoTolerance(), SymmetricTolerance(np.pi, 3)]
        pt = TolEulerPt(pos, Quaternion(), pos_tol, rot_tol)

        tf_samples = pt.sample_grid()
        assert_almost_equal(tf_samples[0], tf_samples[2])
        assert_almost_equal(tf_samples[1][:3, :3], np.eye(3))

    def test_incremental(self):
        pos = [1, 2, 3]
        pos_tol = [NoTolerance()] * 3
        rot_tol = [NoTolerance(), NoTolerance(), SymmetricTolerance(np.pi, 3)]
        pt = TolEulerPt(pos, Quaternion(), pos_tol, rot_tol)

        tf_samples = pt.sample_incremental(10, SampleMethod.random_uniform)

        euler = [rotation_matrix_to_rpy(tf[:3, :3]) for tf in tf_samples]

        for i in range(10):
            assert_almost_equal(euler[i][0], 0)
            assert_almost_equal(euler[i][1], 0)
            assert_in_range(euler[i][2], -np.pi, np.pi)

    def test_transform_to_rel_tolerance_deviation(self):
        tol = [SymmetricTolerance(0.1, 3), Tolerance(0.2, 1.0, 3), NoTolerance()]
        rot_tol = [NoTolerance(), NoTolerance(), SymmetricTolerance(0.1, 3)]
        quat = Quaternion(angle=np.pi / 2, axis=np.array([0, 0, 1]))
        pt = TolEulerPt([1, 2, 3], quat, tol, rot_tol)

        quat2 = Quaternion(angle=np.pi / 2 - 0.05, axis=np.array([0, 0, 1]))
        tf = quat2.transformation_matrix
        tf[:3, 3] = np.array([1.06, 1.91, 3])

        p_rel = pt.transform_to_rel_tolerance_deviation(tf)

        assert p_rel.shape == (6,)

        p_desired = np.array([-0.09, -0.06, 0.0, 0.0, 0.0, -0.05])

        assert_almost_equal(p_rel, p_desired)


class TestTolQuatPt:
    def test_create(self):
        q = Quaternion()
        pos_tol = 3 * [NoTolerance()]
        TolQuatPt([1, 2, 3], q, pos_tol, QuaternionTolerance(0.5))

    def test_sample_incremental(self):
        method = SampleMethod.random_uniform
        q = Quaternion()
        pos_tol = 3 * [NoTolerance()]
        distance = 0.1
        point = TolQuatPt([1, 2, 3], q, pos_tol, QuaternionTolerance(distance))
        samples = point.sample_incremental(100, method)

        for tf in samples:
            newquat = Quaternion(matrix=tf)
            assert Quaternion.distance(q, newquat) <= distance

    def test_to_transform(self):
        distance = 0.1
        q = Quaternion()
        pos_tol = 3 * [NoTolerance()]
        distance = 0.1
        point = TolQuatPt([1, 2, 3], q, pos_tol, QuaternionTolerance(distance))

        tf = point.to_transform([1, 2, 3], q)
        assert_almost_equal(
            [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]], tf
        )


def setting_generator(search_strategy):
    if search_strategy == SearchStrategy.GRID:
        return SamplingSetting(
            SearchStrategy.GRID, 1, SampleMethod.random_uniform, 10, 10, 10, 2
        )
    if search_strategy == SearchStrategy.INCREMENTAL:
        return SamplingSetting(
            SearchStrategy.INCREMENTAL, 1, SampleMethod.random_uniform, 10, 10, 10, 2
        )
    if search_strategy == SearchStrategy.MIN_INCREMENTAL:
        return SamplingSetting(
            SearchStrategy.MIN_INCREMENTAL,
            1,
            SampleMethod.random_uniform,
            10,
            10,
            10,
            2,
        )


def test_to_joint_solutions():
    for search_strategy in SearchStrategy:
        tol = [SymmetricTolerance(0.1, 3), NoTolerance(), NoTolerance()]
        pt = TolPositionPt(
            [1, 2, 3], Quaternion(angle=np.pi / 2, axis=np.array([0, 0, 1])), tol
        )
        robot = DummyRobot(is_colliding=True)
        settings = setting_generator(search_strategy)
        if search_strategy is not SearchStrategy.MIN_INCREMENTAL:
            joint_solutions = pt.to_joint_solutions(robot, settings)
            assert len(joint_solutions) == 0
        else:
            with pytest.raises(Exception) as info:
                joint_solutions = pt.to_joint_solutions(robot, settings)
            msg = "Maximum iterations reached in to_joint_solutions."
            assert msg in str(info.value)
