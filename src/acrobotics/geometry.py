import numpy as np

from acrobotics.shapes import Shape, Box, Polyhedron
from typing import List


class Scene:
    """ Group Shapes and transforms expressed in a common world frame.
    This is used to create more complicated objects that just a box.
    """

    def __init__(self, shapes: List[Shape], tf_shapes: List[np.ndarray]):
        self.s = shapes
        self.tf_s = tf_shapes

    def plot(self, ax, *arg, **kwarg):
        """ Plot all shapes in this geometry.

        With an optional keyword argument "tf" to tell me
        where to plot this shape soup.
        """
        if "tf" in kwarg:
            tf = kwarg.pop("tf")
            for shape, tf_shape in zip(self.s, self.tf_s):
                shape.plot(ax, tf @ tf_shape, *arg, **kwarg)
        else:
            for shape, tf_shape in zip(self.s, self.tf_s):
                shape.plot(ax, tf_shape, *arg, **kwarg)

    @property
    def shapes(self):
        return self.s

    def add(self, shapes: List[Shape], tf_shapes: List[np.ndarray]):
        """ Add new shapes to the scene, inputs are lists. """
        self.s.extend(shapes)
        self.tf_s.extend(tf_shapes)

    def translate(self, x, y, z):
        """ Move all shapes in this scene relative to their current position. """
        for tf in self.tf_s:
            tf[:3, 3] += np.array([x, y, z])

    def get_polyhedrons(self):
        polys = []
        for s, tf in zip(self.s, self.tf_s):
            Ai, bi = s.get_polyhedron(tf)
            polys.append(Polyhedron(Ai, bi))
        return polys

    def is_in_collision(self, other, tf_self=None, tf_other=None):
        tf_shapes_self = self.tf_s
        tf_shapes_other = other.tf_s

        # move the collection of shapes if specified
        if tf_self is not None:
            tf_shapes_self = [tf_self @ tf for tf in tf_shapes_self]
        if tf_other is not None:
            tf_shapes_other = [tf_other @ tf for tf in tf_shapes_other]

        # check for collision between all those shapes
        for tf1, shape1 in zip(tf_shapes_self, self.s):
            for tf2, shape2 in zip(tf_shapes_other, other.s):
                if shape1.is_in_collision(tf1, shape2, tf2):
                    return True
        return False

    def is_path_in_collision(self, tf_self, tf_target, other, tf_other=None):
        tf_shapes_self = [tf_self @ tf for tf in self.tf_s]
        tf_shapes_target = [tf_target @ tf for tf in self.tf_s]

        tf_shapes_other = other.tf_s
        if tf_other is not None:
            tf_shapes_other = [tf_other @ tf for tf in tf_shapes_other]

        # check for collision between all those shapes
        for tf1, tf1_target, shape1 in zip(tf_shapes_self, tf_shapes_target, self.s):
            for tf2, shape2 in zip(tf_shapes_other, other.s):
                if shape1.is_path_in_collision(tf1, tf1_target, shape2, tf2):
                    return True
        return False
