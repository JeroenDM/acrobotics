import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from numpy.testing import assert_almost_equal
from acrobotics.geometry import Scene
from acrobotics.shapes import Box

tf_identity = np.eye(4)
tf_far_away = np.eye(4)
tf_far_away[0, 3] = 10.0


def pose_z(alfa, x, y, z):
    """ Homogenous transform with rotation around z-axis and translation. """
    return np.array(
        [
            [np.cos(alfa), -np.sin(alfa), 0, x],
            [np.sin(alfa), np.cos(alfa), 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1],
        ]
    )


class TestScene:
    def test_polyhedron(self):
        b = Box(1, 2, 3)
        tf = pose_z(0.3, 0.1, 0.2, -0.3)
        col = Scene([b], [tf])
        polys = col.get_polyhedrons()
        assert len(polys) == 1
        Aa = np.array(
            [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
        )
        ba = np.array([0.5, 0.5, 1, 1, 1.5, 1.5])
        Aa = np.dot(Aa, tf[:3, :3].T)
        ba = ba + np.dot(Aa, tf[:3, 3])
        assert_almost_equal(polys[0].A, Aa)
        assert_almost_equal(polys[0].b, ba)

    def test_is_in_collision(self):
        soup_1 = Scene([Box(1, 1, 1)], [tf_identity])
        soup_2 = Scene([Box(1, 1, 2)], [tf_identity])
        assert soup_1.is_in_collision(soup_2) == True

        assert soup_1.is_in_collision(soup_2, tf_self=tf_far_away) == False
        assert soup_1.is_in_collision(soup_2, tf_other=tf_far_away) == False

        soup_3 = Scene(
            [Box(1, 1, 1), Box(0.3, 0.2, 0.1)],
            [pose_z(0.0, -1.0, -1.0, 0.0), -tf_far_away],
        )
        soup_4 = Scene(
            [Box(1, 1, 2), Box(0.3, 0.2, 0.1)],
            [pose_z(np.pi / 4, -2, -2, 0), tf_far_away],
        )
        assert soup_3.is_in_collision(soup_4) == False
        assert soup_4.is_in_collision(soup_3) == False

    def test_plot(self):
        tf2 = pose_z(np.pi / 6, 9, 0, 0)
        soup = Scene([Box(1, 1, 1), Box(2, 2, 0.5)], [tf_identity, tf2])

        fig = plt.figure()
        ax = fig.gca(projection="3d")
        ax.set_xlim([0, 10])
        ax.set_ylim([-5, 5])
        ax.set_zlim([-5, 5])
        soup.plot(ax, c="g")

        tf3 = pose_z(-np.pi / 4, 0, 3, 0)
        soup.plot(ax, tf=tf3, c="r")
        # plt.show(block=True)
