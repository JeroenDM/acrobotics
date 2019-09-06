import fcl
import numpy as np

from abc import ABC, abstractmethod
from collections import namedtuple


def transform_vector(tf, vector):
    """ Transform a vector with a homogeneous transform matrix tf. """
    return np.dot(tf[:3, :3], vector) + tf[:3, 3]


Polyhedron = namedtuple("Polyhedron", ["A", "b"])


class Shape(ABC):
    """ Shape for visualization and collision checking.

    Wraps around an fcl_shape for collision checking.
    Generated vertices and edges for plotting.

    A Shape has no inherent position! You always have to specify
    a transform when you want something from a shape.
    If you want fixed shapes for a planning scene,
    you need to use a Scene object that contains a shape and a transform.

    """

    fcl_shape: fcl.CollisionGeometry
    request: fcl.CollisionRequest
    result: fcl.CollisionResult

    @abstractmethod
    def get_vertices(self, transform: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_edges(self, transform: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_normals(self, transform: np.ndarray) -> np.ndarray:
        pass

    def is_in_collision(self, tf, other, tf_other):
        """ Collision checking with another shape for the given transforms. """
        fcl_tf_1 = fcl.Transform(tf[:3, :3], tf[:3, 3])
        fcl_tf_2 = fcl.Transform(tf_other[:3, :3], tf_other[:3, 3])

        o1 = fcl.CollisionObject(self.fcl_shape, fcl_tf_1)
        o2 = fcl.CollisionObject(other.fcl_shape, fcl_tf_2)

        return fcl.collide(o1, o2, self.request, self.result)

    def get_empty_plot_lines(self, ax, *arg, **kwarg):
        """ Create empty lines to initialize an animation """
        return [ax.plot([], [], "-", *arg, **kwarg)[0] for i in range(12)]

    def update_plot_lines(self, lines, tf):
        """ Update existing lines on a plot using the given transform tf"""
        edges = self.get_edges(tf)
        for i, l in enumerate(lines):
            x = [edges[i, 0], edges[i, 3]]
            y = [edges[i, 1], edges[i, 4]]
            z = [edges[i, 2], edges[i, 5]]
            l.set_data(x, y)
            l.set_3d_properties(z)
        return lines

    def plot(self, ax, tf, *arg, **kwarg):
        """ Plot a box as lines on a given axes_handle."""
        lines = self.get_empty_plot_lines(ax, *arg, **kwarg)
        lines = self.update_plot_lines(lines, tf)


class Box(Shape):
    """
    I'm just a Box with six sides.
    But you would be suprised,
    how many robots this provides.
    """

    def __init__(self, dx, dy, dz):
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.fcl_shape = fcl.Box(dx, dy, dz)
        self.request = fcl.CollisionRequest()
        self.result = fcl.CollisionResult()

    def get_vertices(self, tf):
        v = np.zeros((8, 3))
        a = self.dx / 2
        b = self.dy / 2
        c = self.dz / 2

        v[0] = transform_vector(tf, [-a, b, c])
        v[1] = transform_vector(tf, [-a, b, -c])
        v[2] = transform_vector(tf, [-a, -b, c])
        v[3] = transform_vector(tf, [-a, -b, -c])

        v[4] = transform_vector(tf, [a, b, c])
        v[5] = transform_vector(tf, [a, b, -c])
        v[6] = transform_vector(tf, [a, -b, c])
        v[7] = transform_vector(tf, [a, -b, -c])
        return v

    def get_edges(self, tf):
        v = self.get_vertices(tf)
        e = np.zeros((12, 6))
        e[0] = np.hstack((v[0], v[1]))
        e[1] = np.hstack((v[1], v[3]))
        e[2] = np.hstack((v[3], v[2]))
        e[3] = np.hstack((v[2], v[0]))

        e[4] = np.hstack((v[0], v[4]))
        e[5] = np.hstack((v[1], v[5]))
        e[6] = np.hstack((v[3], v[7]))
        e[7] = np.hstack((v[2], v[6]))

        e[8] = np.hstack((v[4], v[5]))
        e[9] = np.hstack((v[5], v[7]))
        e[10] = np.hstack((v[7], v[6]))
        e[11] = np.hstack((v[6], v[4]))
        return e

    def get_normals(self, tf):
        n = np.zeros((6, 3))
        R = tf[:3, :3]
        n[0] = np.dot(R, [1, 0, 0])
        n[1] = np.dot(R, [-1, 0, 0])
        n[2] = np.dot(R, [0, 1, 0])
        n[3] = np.dot(R, [0, -1, 0])
        n[4] = np.dot(R, [0, 0, 1])
        n[5] = np.dot(R, [0, 0, -1])
        return n

    def get_polyhedron(self, tf):
        """ Shape represented as inequality A*x <= b
        
        This is usefull when modelling the "no collision" constraints
        as a separating hyperplane problem.
        """
        A = self.get_normals(tf)
        b = 0.5 * np.array([self.dx, self.dx, self.dy, self.dy, self.dz, self.dz])
        b = b + np.dot(A, tf[:3, 3])
        return Polyhedron(A, b)
