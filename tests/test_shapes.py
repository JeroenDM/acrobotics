import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from numpy.testing import assert_almost_equal
from acrolib.geometry import rot_z, rot_y

from acrobotics.shapes import Box, Cylinder


tf_identity = np.eye(4)


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


class TestShape:
    def test_init(self):
        Box(1, 2, 3)

    def test_get_vertices(self):
        b = Box(1, 2, 3)
        v = b.get_vertices(tf_identity)
        desired = np.array(
            [
                [-0.5, 1, 1.5],
                [-0.5, 1, -1.5],
                [-0.5, -1, 1.5],
                [-0.5, -1, -1.5],
                [0.5, 1, 1.5],
                [0.5, 1, -1.5],
                [0.5, -1, 1.5],
                [0.5, -1, -1.5],
            ]
        )
        assert_almost_equal(v, desired)

    def test_get_normals(self):
        b = Box(1, 2, 3)
        n = b.get_normals(tf_identity)
        desired = np.array(
            [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
        )
        assert_almost_equal(n, desired)

    def test_set_transform(self):
        b = Box(1, 2, 3)
        tf = np.eye(4)
        tf[0, 3] = 10.5
        v = b.get_vertices(tf)
        desired = np.array(
            [
                [10, 1, 1.5],
                [10, 1, -1.5],
                [10, -1, 1.5],
                [10, -1, -1.5],
                [11, 1, 1.5],
                [11, 1, -1.5],
                [11, -1, 1.5],
                [11, -1, -1.5],
            ]
        )
        assert_almost_equal(v, desired)

    def test_set_transform2(self):
        b = Box(1, 2, 3)
        tf = np.eye(4)
        # rotate pi / 2 around x-axis
        tf[1:3, 1:3] = np.array([[0, -1], [1, 0]])
        v = b.get_vertices(tf)
        desired = np.array(
            [
                [-0.5, -1.5, 1],
                [-0.5, 1.5, 1],
                [-0.5, -1.5, -1],
                [-0.5, 1.5, -1],
                [0.5, -1.5, 1],
                [0.5, 1.5, 1],
                [0.5, -1.5, -1],
                [0.5, 1.5, -1],
            ]
        )
        assert_almost_equal(v, desired)

    def test_get_edges(self):
        b = Box(1, 2, 3)
        e = b.get_edges(tf_identity)
        row, col = e.shape
        assert row == 12
        assert col == 6
        v = b.get_vertices(tf_identity)
        # check only one edge
        v0 = np.hstack((v[0], v[1]))
        assert_almost_equal(v0, e[0])

    def test_polyhedron(self):
        b = Box(1, 2, 3)
        A, b = b.get_polyhedron(np.eye(4))
        Aa = np.array(
            [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
        )
        ba = np.array([0.5, 0.5, 1, 1, 1.5, 1.5])
        assert_almost_equal(A, Aa)
        assert_almost_equal(b, ba)

    def test_polyhedron_transformed(self):
        b = Box(1, 2, 3)
        tf = pose_z(0.3, 0.1, 0.2, -0.3)
        A, b = b.get_polyhedron(tf)
        Aa = np.array(
            [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
        )
        ba = np.array([0.5, 0.5, 1, 1, 1.5, 1.5])
        Aa = np.dot(Aa, tf[:3, :3].T)
        ba = ba + np.dot(Aa, tf[:3, 3])
        assert_almost_equal(A, Aa)
        assert_almost_equal(b, ba)

    def test_is_in_collision(self):
        b1 = Box(1, 1, 1)
        b2 = Box(1, 1, 2)
        actual = b1.is_in_collision(tf_identity, b2, tf_identity)
        assert actual == True

        b3 = Box(1, 2, 1)
        T3 = pose_z(np.pi / 4, 0.7, 0.7, 0)
        assert b1.is_in_collision(tf_identity, b3, T3) == True

        b4 = Box(1, 1, 1)
        b5 = Box(1, 1, 2)
        T4 = pose_z(0, -1, -1, 0)
        T5 = pose_z(np.pi / 4, -2, -2, 0)
        assert b4.is_in_collision(T4, b5, T5) == False

    def test_plot(self):
        b1 = Box(1, 2, 3)
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        b1.plot(ax, tf_identity)
        assert True


class TestCylinder:
    def test_4_faces(self):
        cyl = Cylinder(1, 2, approx_faces=4)

        n = cyl.get_normals(np.eye(4))
        n_desired = np.array(
            [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
        )
        assert_almost_equal(n, n_desired)

        v = cyl.get_vertices(np.eye(4))
        v_desired = np.array(
            [
                [1, 0, 1],
                [0, 1, 1],
                [-1, 0, 1],
                [0, -1, 1],
                [1, 0, -1],
                [0, 1, -1],
                [-1, 0, -1],
                [0, -1, -1],
            ]
        )
        v_desired = (rot_z(np.pi / 4) @ v_desired.T).T
        assert_almost_equal(v, v_desired)

        e = cyl.get_edges(np.eye(4))
        e_desired = np.zeros((12, 6))

        vd = v_desired
        e_desired[0] = np.hstack((vd[3], vd[0]))
        e_desired[1] = np.hstack((vd[0], vd[1]))
        e_desired[2] = np.hstack((vd[1], vd[2]))
        e_desired[3] = np.hstack((vd[2], vd[3]))

        e_desired[4] = np.hstack((vd[7], vd[4]))
        e_desired[5] = np.hstack((vd[4], vd[5]))
        e_desired[6] = np.hstack((vd[5], vd[6]))
        e_desired[7] = np.hstack((vd[6], vd[7]))

        e_desired[8] = np.hstack((vd[0], vd[4]))
        e_desired[9] = np.hstack((vd[1], vd[5]))
        e_desired[10] = np.hstack((vd[2], vd[6]))
        e_desired[11] = np.hstack((vd[3], vd[7]))

        assert e.shape == e_desired.shape
        assert_almost_equal(e[0:4], e_desired[0:4])
        assert_almost_equal(e[4:8], e_desired[4:8])
        assert_almost_equal(e[8:12], e_desired[8:12])

    def test_4_faces_transformed(self):

        tf = np.eye(4)
        tf[:3, 3] = np.array([5, -3, 7])
        tf[:3, :3] = rot_y(0.5) @ rot_z(-0.3)

        cyl = Cylinder(1, 2, approx_faces=4)

        n = cyl.get_normals(tf)
        n_desired = np.array(
            [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
        )
        n_desired = (tf[:3, :3] @ n_desired.T).T
        assert_almost_equal(n, n_desired)

        v = cyl.get_vertices(tf)
        v_desired = np.array(
            [
                [1, 0, 1],
                [0, 1, 1],
                [-1, 0, 1],
                [0, -1, 1],
                [1, 0, -1],
                [0, 1, -1],
                [-1, 0, -1],
                [0, -1, -1],
            ]
        )
        v_desired = (rot_z(np.pi / 4) @ v_desired.T).T
        v_desired = (tf[:3, :3] @ v_desired.T).T + tf[:3, 3]
        assert_almost_equal(v, v_desired)

    def test_plot_cylinder(self):
        cyl = Cylinder(1, 2)

        fig = plt.figure()
        ax = fig.gca(projection="3d")

        tf = np.eye(4)
        tf[:3, 3] = np.array([0, 5, -3])
        tf[:3, :3] = rot_y(np.pi / 4)

        cyl.plot(ax, tf, c="k")

        # plt.show(block=True)
        assert True
