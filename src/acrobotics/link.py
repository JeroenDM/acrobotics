import numpy as np
import casadi as ca

from collections import namedtuple
from enum import Enum


DHLink = namedtuple("DHLink", ["a", "alpha", "d", "theta"])


class JointType(Enum):
    revolute = "r"
    prismatic = "p"


class LinkKinematics:
    """ Robot link according to the Denavit-Hartenberg convention. """

    def __init__(self, dh_parameters: DHLink, joint_type: JointType):
        """ Creates a linkf from Denavit-Hartenberg parameters,
        a joint type ('r' for revolute, 'p' for prismatic) and
        a Scene of Shapes representing the geometry.
        """
        self.dh = dh_parameters

        if joint_type in JointType:
            self.joint_type = joint_type
        else:
            raise ValueError(f"Unkown JointType: {joint_type}.")

        # chache a transform because creating it is slow
        # but just fillin in an existing one is ok
        self._T = np.eye(4)

    def get_link_relative_transform(self, qi):
        """ transformation matrix from link i relative to i-1

        Links and joints are numbered from 1 to ndof, but python
        indexing of these links goes from 0 to ndof-1!
        """
        if self.joint_type == JointType.revolute:
            a, alpha, d, theta = self.dh.a, self.dh.alpha, self.dh.d, qi
        elif self.joint_type == JointType.prismatic:
            a, alpha, d, theta = self.dh.a, self.dh.alpha, qi, self.dh.theta

        c_theta = np.cos(theta)
        s_theta = np.sin(theta)
        c_alpha = np.cos(alpha)
        s_alpha = np.sin(alpha)
        T = self._T
        T[0, 0], T[0, 1] = c_theta, -s_theta * c_alpha
        T[0, 2], T[0, 3] = s_theta * s_alpha, a * c_theta

        T[1, 0], T[1, 1] = s_theta, c_theta * c_alpha
        T[1, 2], T[1, 3] = -c_theta * s_alpha, a * s_theta

        T[2, 1], T[2, 2], T[2, 3] = s_alpha, c_alpha, d
        return T

    def get_link_relative_transform_casadi(self, qi):
        """ Link transform according to the Denavit-Hartenberg convention.
        Casadi compatible function.
        """
        if self.joint_type == JointType.revolute:
            a, alpha, d, theta = self.dh.a, self.dh.alpha, self.dh.d, qi
        elif self.joint_type == JointType.prismatic:
            a, alpha, d, theta = self.dh.a, self.dh.alpha, qi, self.dh.theta

        c_t, s_t = ca.cos(theta), ca.sin(theta)
        c_a, s_a = ca.cos(alpha), ca.sin(alpha)

        row1 = ca.hcat([c_t, -s_t * c_a, s_t * s_a, a * c_t])
        row2 = ca.hcat([s_t, c_t * c_a, -c_t * s_a, a * s_t])
        row3 = ca.hcat([0, s_a, c_a, d])
        row4 = ca.hcat([0, 0, 0, 1])

        return ca.vcat([row1, row2, row3, row4])


class Link(LinkKinematics):
    def __init__(self, dh_parameters, joint_type, geometry):
        super().__init__(dh_parameters, joint_type)
        self.geometry = geometry

    def plot(self, ax, tf, *arg, **kwarg):
        self.geometry.plot(ax, tf=tf, *arg, **kwarg)
