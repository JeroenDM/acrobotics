import numpy as np

from acrolib.geometry import pose_x, translation

from acrobotics.link import Link, DHLink, JointType
from acrobotics.robot import Robot, JointLimit
from acrobotics.shapes import Box
from acrobotics.geometry import Scene


class KukaIIWA(Robot):
    def __init__(self):
        pi2 = np.pi / 2
        # define kuka collision shapes
        s = [
            Box(0.2, 0.3, 0.1),
            Box(0.1, 0.1, 0.4),
            Box(0.4, 0.1, 0.1),
            Box(0.1, 0.1, 0.1),
            Box(0.1, 0.1, 0.1),
            Box(0.1, 0.1, 0.1),
            Box(0.1, 0.1, 0.1),
        ]
        # define transforms for collision shapes
        tfs = [
            translation(0, -0.15, -0.05),
            pose_x(0, 0, 0, 0),
            pose_x(0, 0, 0, 0),
            pose_x(0, 0, 0, 0),
            pose_x(0, 0, 0, 0),
            pose_x(0, 0, 0, 0),
            pose_x(0, 0, 0, 0),
        ]

        # Denavit-Hartenberg a, alpha, d, theta
        dh_links = [
            DHLink(0, pi2, 0.360, 0),
            DHLink(0, -pi2, 0, 0),
            DHLink(0, -pi2, 0.420, 0),
            DHLink(0, pi2, 0, 0),
            DHLink(0, pi2, 0.400, 0),
            DHLink(0, -pi2, 0, 0),
            DHLink(0, 0, 0.125, 0),
        ]

        geoms = [Scene([shape], [tf]) for shape, tf in zip(s, tfs)]

        jls = [
            JointLimit(np.deg2rad(-170), np.deg2rad(170)),
            JointLimit(np.deg2rad(-120), np.deg2rad(120)),
            JointLimit(np.deg2rad(-170), np.deg2rad(170)),
            JointLimit(np.deg2rad(-120), np.deg2rad(120)),
            JointLimit(np.deg2rad(-170), np.deg2rad(170)),
            JointLimit(np.deg2rad(-120), np.deg2rad(120)),
            JointLimit(np.deg2rad(-175), np.deg2rad(175)),
        ]

        # create robot
        super().__init__(
            [Link(dh_links[i], JointType.revolute, geoms[i]) for i in range(6)], jls
        )

        # self.geometry_base = Scene(
        #     [Box(0.15, 0.15, 0.15)], [translation(0, 0, -0.36 - 0.15 / 2)]
        # )


import matplotlib.pyplot as plt
from acrolib.plotting import get_default_axes3d

robot = KukaIIWA()

fig, ax = get_default_axes3d()

robot.plot(ax, [0, 0, 0, 0, 0, 0, 0], c="k")
robot.plot_kinematics(ax, [0] * 7)

plt.show()
