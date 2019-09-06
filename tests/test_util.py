import numpy as np
import matplotlib
import mpl_toolkits

from numpy.testing import assert_almost_equal
from acrobotics.util import (
    rot_x,
    rot_y,
    rot_z,
    pose_x,
    pose_y,
    pose_z,
    get_default_axes3d,
    plot_reference_frame,
)


class TestMatrixCreation:
    def test_rot_x(self):
        A = rot_x(0.6)
        B = rot_x(-0.6)
        assert_almost_equal(A @ B, np.eye(3))
        assert_almost_equal(A[0, :3], np.array([1, 0, 0]))
        assert_almost_equal(A[:3, 0], np.array([1, 0, 0]))

    def test_rot_y(self):
        A = rot_y(0.6)
        B = rot_y(-0.6)
        assert_almost_equal(A @ B, np.eye(3))
        assert_almost_equal(A[1, :3], np.array([0, 1, 0]))
        assert_almost_equal(A[:3, 1], np.array([0, 1, 0]))

    def test_rot_z(self):
        A = rot_z(0.6)
        B = rot_z(-0.6)
        assert_almost_equal(A @ B, np.eye(3))
        assert_almost_equal(A[2, :3], np.array([0, 0, 1]))
        assert_almost_equal(A[:3, 2], np.array([0, 0, 1]))

    def test_pose_x(self):
        A = pose_x(0.6, 1, 2, 3)
        assert_almost_equal(A[0, :3], np.array([1, 0, 0]))
        assert_almost_equal(A[:3, 0], np.array([1, 0, 0]))
        assert_almost_equal(A[:3, 3], np.array([1, 2, 3]))
        assert_almost_equal(A[3, :], np.array([0, 0, 0, 1]))

    def test_pose_y(self):
        A = pose_y(0.6, 1, 2, 3)
        assert_almost_equal(A[1, :3], np.array([0, 1, 0]))
        assert_almost_equal(A[:3, 1], np.array([0, 1, 0]))
        assert_almost_equal(A[:3, 3], np.array([1, 2, 3]))
        assert_almost_equal(A[3, :], np.array([0, 0, 0, 1]))

    def test_pose_z(self):
        A = pose_z(0.6, 1, 2, 3)
        assert_almost_equal(A[2, :3], np.array([0, 0, 1]))
        assert_almost_equal(A[:3, 2], np.array([0, 0, 1]))
        assert_almost_equal(A[:3, 3], np.array([1, 2, 3]))
        assert_almost_equal(A[3, :], np.array([0, 0, 0, 1]))


class TestPlotUtil:
    def test_create_axes_3d(self):
        fig, ax = get_default_axes3d()
        assert isinstance(fig, matplotlib.pyplot.Figure)
        assert isinstance(ax, mpl_toolkits.mplot3d.Axes3D)

    def test_plot_reference_frame(self):
        _, ax = get_default_axes3d()
        plot_reference_frame(ax)
        plot_reference_frame(ax, tf=np.eye(4))
        plot_reference_frame(ax, tf=np.eye(4), arrow_length=0.3)
