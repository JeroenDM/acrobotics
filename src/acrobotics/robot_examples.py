import numpy as np

from .robot import Robot, Tool
from .link import Link, DHLink, JointType
from .geometry import ShapeSoup
from .shapes import Box
from .util import translation, pose_x
from .inverse_kinematics import spherical_wrist, anthropomorphic_arm, arm_2
from .inverse_kinematics.ik_result import IKResult

PI = np.pi


def tf_inverse(T):
    """ Efficient inverse of a homogenous transform.

    (Normal matrix inversion would be a bad idea.)
    Returns a copy, not inplace!
    """
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].transpose()
    Ti[:3, 3] = np.dot(-Ti[:3, :3], T[:3, 3])
    return Ti


class PlanarArm(Robot):
    """ Robot defined on page 69 in book Siciliano """

    def __init__(self, a1=1, a2=1, a3=1):
        geometry = [
            ShapeSoup([Box(-a1 / 2, 0.1, 0.1)], [np.eye(4)]),
            ShapeSoup([Box(-a2 / 2, 0.1, 0.1)], [np.eye(4)]),
            ShapeSoup([Box(-a3 / 2, 0.1, 0.1)], [np.eye(4)]),
        ]
        super().__init__(
            [
                Link(DHLink(a1, 0, 0, 0), JointType.revolute, geometry[0]),
                Link(DHLink(a2, 0, 0, 0), JointType.revolute, geometry[1]),
                Link(DHLink(a3, 0, 0, 0), JointType.revolute, geometry[2]),
            ]
        )


class SphericalArm(Robot):
    """ Robot defined on page 72 in book Siciliano """

    def __init__(self, d2=1):
        geometry = [
            ShapeSoup([Box(1, 0.1, 0.1)], [np.eye(4)]),
            ShapeSoup([Box(1, 0.1, 0.1)], [np.eye(4)]),
            ShapeSoup([Box(1, 0.1, 0.1)], [np.eye(4)]),
        ]
        super().__init__(
            [
                Link(DHLink(0, -PI / 2, 0, 0), JointType.revolute, geometry[0]),
                Link(DHLink(0, PI / 2, d2, 0), JointType.revolute, geometry[1]),
                Link(DHLink(0, 0, 0, 0), JointType.prismatic, geometry[2]),
            ]
        )


class SphericalWrist(Robot):
    """ Robot defined on page 75 in book Siciliano """

    def __init__(self, d3=1):
        geometry = [
            ShapeSoup([Box(1, 0.1, 0.1)], [np.eye(4)]),
            ShapeSoup([Box(1, 0.1, 0.1)], [np.eye(4)]),
            ShapeSoup([Box(1, 0.1, 0.1)], [np.eye(4)]),
        ]
        super().__init__(
            [
                Link(DHLink(0, -PI / 2, 0, 0), JointType.revolute, geometry[0]),
                Link(DHLink(0, PI / 2, 0, 0), JointType.revolute, geometry[1]),
                Link(DHLink(0, 0, d3, 0), JointType.revolute, geometry[2]),
            ]
        )

        # cache ik solution array object for performance
        # self.qsol = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    def ik(self, T):
        return spherical_wrist.ik(T, self.tf_base)


class AnthropomorphicArm(Robot):
    """ Robot defined on page 73 in book Siciliano """

    def __init__(self, a2=1, a3=1):
        geometry = [
            ShapeSoup([Box(0.1, 0.3, 0.3)], [translation(-0.05, 0, 0)]),
            ShapeSoup([Box(a2, 0.1, 0.1)], [translation(-a2 / 2, 0, 0)]),
            ShapeSoup([Box(a3, 0.1, 0.1)], [translation(-a3 / 2, 0, 0)]),
        ]
        super().__init__(
            [
                Link(DHLink(0, PI / 2, 0, 0), JointType.revolute, geometry[0]),
                Link(DHLink(a2, 0, 0, 0), JointType.revolute, geometry[1]),
                Link(DHLink(a3, 0, 0, 0), JointType.revolute, geometry[2]),
            ]
        )

    def ik(self, T):
        return anthropomorphic_arm.ik(T, self.links)


class Arm2(Robot):
    """ Articulated arm with first link length is NOT zeros
    In addition the last frame is rotated to get the Z axis pointing out
    along the hypothetical 4 joint when adding a wrist"""

    def __init__(self, a1=1, a2=1, a3=1):
        geometry = [
            ShapeSoup([Box(a1, 0.1, 0.1)], [np.eye(4)]),
            ShapeSoup([Box(a2, 0.1, 0.1)], [np.eye(4)]),
            ShapeSoup([Box(a3, 0.1, 0.1)], [np.eye(4)]),
        ]
        super().__init__(
            [
                Link(DHLink(a1, PI / 2, 0, 0), JointType.revolute, geometry[0]),
                Link(DHLink(a2, 0, 0, 0), JointType.revolute, geometry[1]),
                Link(DHLink(a3, PI / 2, 0, 0), JointType.revolute, geometry[2]),
            ]
        )

    def ik(self, T):
        return arm_2.ik(T, self.links)


class Kuka(Robot):
    """ Robot combining AnthropomorphicArm and SphericalWrist
    """

    def __init__(self, a1=0.18, a2=0.6, d4=0.62, d6=0.115):
        # define kuka collision shapes
        s = [
            Box(0.3, 0.2, 0.1),
            Box(0.8, 0.2, 0.1),
            Box(0.2, 0.1, 0.5),
            Box(0.1, 0.2, 0.1),
            Box(0.1, 0.1, 0.085),
            Box(0.1, 0.1, 0.03),
        ]
        # define transforms for collision shapes
        tfs = [
            pose_x(0, -0.09, 0, 0.05),
            pose_x(0, -0.3, 0, -0.05),
            pose_x(0, 0, 0.05, 0.17),
            pose_x(0, 0, 0.1, 0),
            pose_x(0, 0, 0, 0.085 / 2),
            pose_x(0, 0, 0, -0.03 / 2),
        ]

        dh_links = [
            DHLink(a1, PI / 2, 0, 0),
            DHLink(a2, 0, 0, 0),
            DHLink(0, PI / 2, 0, 0),
            DHLink(0, -PI / 2, d4, 0),
            DHLink(0, PI / 2, 0, 0),
            DHLink(0, 0, d6, 0),
        ]

        geoms = [ShapeSoup([shape], [tf]) for shape, tf in zip(s, tfs)]

        # create robot
        super().__init__(
            [Link(dh_links[i], JointType.revolute, geoms[i]) for i in range(6)]
        )
        self.arm = Arm2(a1=a1, a2=a2, a3=d4)
        self.wrist = SphericalWrist(d3=d6)

    def ik(self, T) -> IKResult:
        # copy transform to change it without affecting the T given
        Tw = T.copy()
        # compensate for base
        Tw = np.dot(tf_inverse(self.tf_base), Tw)
        # compensate for tool frame
        if self.tf_tool_tip is not None:
            Tw = np.dot(Tw, tf_inverse(self.tf_tool_tip))
        # compensate for d6, last link length
        d6 = self.links[5].dh.d
        v6 = np.dot(Tw[:3, :3], np.array([0, 0, d6]))
        Tw[:3, 3] = Tw[:3, 3] - v6
        sol_arm = self.arm.ik(Tw)
        if sol_arm.success:
            solutions = []
            for q_arm in sol_arm.solutions:
                q_arm[2] = q_arm[2] + PI / 2
                base_wrist = np.eye(4)
                # get orientation from arm fk
                base_wrist[:3, :3] = self.arm.fk(q_arm)[:3, :3]
                # position from given T (not used actually)
                base_wrist[3, :3] = Tw[3, :3]
                self.wrist.tf_base = base_wrist
                sol_wrist = self.wrist.ik(Tw)
                if sol_wrist.success:
                    for q_wrist in sol_wrist.solutions:
                        solutions.append(np.hstack((q_arm, q_wrist)))
            if len(solutions) > 0:
                return IKResult(True, solutions)
        return IKResult(False)


class KukaOnRail(Robot):
    def __init__(self, a1=0.18, a2=0.6, d4=0.62, d6=0.115):
        s = [
            Box(0.2, 0.2, 0.1),
            Box(0.3, 0.2, 0.1),
            Box(0.8, 0.2, 0.1),
            Box(0.2, 0.1, 0.5),
            Box(0.1, 0.2, 0.1),
            Box(0.1, 0.1, 0.085),
            Box(0.1, 0.1, 0.03),
        ]

        tfs = [
            pose_x(0, 0, 0, -0.15),
            pose_x(0, -0.09, 0, 0.05),
            pose_x(0, -0.3, 0, -0.05),
            pose_x(0, 0, 0.05, 0.17),
            pose_x(0, 0, 0.1, 0),
            pose_x(0, 0, 0, 0.085 / 2),
            pose_x(0, 0, 0, -0.03 / 2),
        ]
        # create robot
        super().__init__(
            [
                Link(
                    DHLink(0, PI / 2, 0, 0),
                    JointType.prismatic,
                    ShapeSoup([s[0]], [tfs[0]]),
                ),
                Link(
                    DHLink(a1, PI / 2, 0, 0),
                    JointType.revolute,
                    ShapeSoup([s[1]], [tfs[1]]),
                ),
                Link(
                    DHLink(a2, 0, 0, 0), JointType.revolute, ShapeSoup([s[2]], [tfs[2]])
                ),
                Link(
                    DHLink(0, PI / 2, 0, 0),
                    JointType.revolute,
                    ShapeSoup([s[3]], [tfs[3]]),
                ),
                Link(
                    DHLink(0, -PI / 2, d4, 0),
                    JointType.revolute,
                    ShapeSoup([s[4]], [tfs[4]]),
                ),
                Link(
                    DHLink(0, PI / 2, 0, 0),
                    JointType.revolute,
                    ShapeSoup([s[5]], [tfs[5]]),
                ),
                Link(
                    DHLink(0, 0, d6, 0), JointType.revolute, ShapeSoup([s[6]], [tfs[6]])
                ),
            ]
        )
        self.kuka = Kuka(a1=0.18, a2=0.6, d4=0.62, d6=0.115)

    def ik(self, T, q_fixed) -> IKResult:
        # copy transform to change it without affecting the T given
        Tw = T.copy()
        # compensate for base
        Tw = np.dot(tf_inverse(self.tf_base), Tw)
        # compensate for tool frame
        if self.tf_tool_tip is not None:
            Tw = np.dot(Tw, tf_inverse(self.tf_tool_tip))

        # change base of helper robot according to fixed joint
        T1 = self.links[0].get_link_relative_transform(q_fixed)
        self.kuka.tf_base = T1

        res = self.kuka.ik(Tw)
        if res.success:
            q_sol = []
            for qi in res.solutions:
                q_sol.append(np.hstack((q_fixed, qi)))
            return IKResult(True, q_sol)
        else:
            return IKResult(False)
