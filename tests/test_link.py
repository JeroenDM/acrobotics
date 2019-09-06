from acrobotics.link import Link, LinkKinematics, DHLink, JointType

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from acrobotics.geometry import ShapeSoup
from acrobotics.shapes import Box
from numpy.testing import assert_almost_equal


def DenHarMat(theta, alpha, a, d):
    """ Use code from someone else to compare with:
    https://stackoverflow.com/questions/17891024/forward-kinematics-data-modeling
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)

    return np.array(
        [
            [cos_theta, -sin_theta * cos_alpha, sin_theta * sin_alpha, a * cos_theta],
            [sin_theta, cos_theta * cos_alpha, -cos_theta * sin_alpha, a * sin_theta],
            [0, sin_alpha, cos_alpha, d],
            [0, 0, 0, 1],
        ]
    )


class TestLinkKinematics:
    def test_init(self):
        dh_params = DHLink(0.1, np.pi / 4, -0.1, np.pi / 6)
        link1 = LinkKinematics(dh_params, JointType.revolute)
        link2 = LinkKinematics(dh_params, JointType.prismatic)

        assert link1.joint_type == JointType.revolute
        assert link2.joint_type == JointType.prismatic
        assert link1.dh == dh_params
        assert link2.dh == dh_params

    def test_dh_matrix(self):
        dh_params = DHLink(0.1, np.pi / 4, -0.1, np.pi / 6)
        link1 = LinkKinematics(dh_params, JointType.revolute)
        link2 = LinkKinematics(dh_params, JointType.prismatic)

        q1 = 1.2
        T1 = link1.get_link_relative_transform(q1)
        T1_desired = DenHarMat(q1, dh_params.alpha, dh_params.a, dh_params.d)
        assert_almost_equal(T1, T1_desired)

        d2 = 0.75
        T2 = link2.get_link_relative_transform(d2)
        T2_desired = DenHarMat(dh_params.theta, dh_params.alpha, dh_params.a, d2)
        assert_almost_equal(T2, T2_desired)

    def test_dh_matrix_casadi(self):
        dh_params = DHLink(0.1, np.pi / 4, -0.1, np.pi / 6)
        link1 = LinkKinematics(dh_params, JointType.revolute)
        link2 = LinkKinematics(dh_params, JointType.prismatic)

        opti = ca.Opti()

        q1 = opti.variable()
        T1 = ca.Function("T1", [q1], [link1.get_link_relative_transform_casadi(q1)])
        T1_desired = DenHarMat(1.2, dh_params.alpha, dh_params.a, dh_params.d)
        assert_almost_equal(np.array(T1(1.2)), T1_desired)

        d1 = opti.variable()
        T2 = ca.Function("T2", [d1], [link2.get_link_relative_transform_casadi(d1)])
        T2_desired = DenHarMat(dh_params.theta, dh_params.alpha, dh_params.a, 0.75)
        assert_almost_equal(np.array(T2(0.75)), T2_desired)


class TestLink:
    def test_init(self):
        dh_params = DHLink(0.1, np.pi / 4, -0.1, np.pi / 6)
        geometry = ShapeSoup([Box(1, 2, 3)], [np.eye(4)])
        link1 = Link(dh_params, JointType.revolute, geometry)

        fig = plt.figure()
        ax = fig.gca(projection="3d")
        link1.plot(ax, np.eye(4))
