import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from numpy.testing import assert_almost_equal
from acrobotics.robot_examples import (
    PlanarArm,
    SphericalArm,
    AnthropomorphicArm,
    Arm2,
    Kuka,
    KukaOnRail,
)
from .fk_implementations import FKImplementations as fki
from acrobotics.util import pose_x
from acrobotics.tool_examples import torch

PI = np.pi


robot1 = PlanarArm(a1=1, a2=1.5, a3=0.5)
robot2 = SphericalArm(d2=2.6)
robot3 = AnthropomorphicArm()
robot4 = Arm2()


class TestForwardKinematics:
    def generate_random_configurations(self, robot, N=5):
        C = []
        for jl in robot.joint_limits:
            C.append(np.random.uniform(jl.lower, jl.upper, size=N))
        return np.vstack(C).T

    def test_PlanarArm_robot(self):
        q_test = self.generate_random_configurations(robot1)
        for qi in q_test:
            Tactual = robot1.fk(qi)
            Tdesired = fki.fk_PlanarArm(qi, robot1.links)
            assert_almost_equal(Tactual, Tdesired)

    def test_SphericalArm_robot(self):
        q_test = self.generate_random_configurations(robot2)
        for qi in q_test:
            Tactual = robot2.fk(qi)
            Tdesired = fki.fk_SphericalArm(qi, robot2.links)
            assert_almost_equal(Tactual, Tdesired)

    def test_AnthropomorphicArm_robot(self):
        q_test = self.generate_random_configurations(robot3)
        for qi in q_test:
            Tactual = robot3.fk(qi)
            Tdesired = fki.fk_AnthropomorphicArm(qi, robot3.links)
            assert_almost_equal(Tactual, Tdesired)

    def test_Arm2_robot(self):
        q_test = self.generate_random_configurations(robot4)
        for qi in q_test:
            Tactual = robot4.fk(qi)
            Tdesired = fki.fk_Arm2(qi, robot4.links)
            assert_almost_equal(Tactual, Tdesired)

    def test_Arm2_tool_robot(self):
        bot = Arm2()
        bot.tf_tool_tip = torch.tf_tool_tip
        q_test = self.generate_random_configurations(bot)
        for qi in q_test:
            Tactual = bot.fk(qi)
            Tdesired = fki.fk_Arm2(qi, bot.links)
            Tdesired = Tdesired @ torch.tf_tool_tip
            assert_almost_equal(Tactual, Tdesired)

    def test_Kuka_robot(self):
        bot = Kuka()
        q_test = self.generate_random_configurations(bot)
        for qi in q_test:
            Tactual = bot.fk(qi)
            Tdesired = fki.fk_kuka(qi)
            assert_almost_equal(Tactual, Tdesired)

    def test_Kuka_tool_robot(self):
        bot = Kuka()
        bot.tf_tool_tip = torch.tf_tool_tip
        q_test = self.generate_random_configurations(bot)
        for qi in q_test:
            Tactual = bot.fk(qi)
            Tdesired = fki.fk_kuka(qi)
            Tdesired = Tdesired @ torch.tf_tool_tip
            assert_almost_equal(Tactual, Tdesired)

    def test_Kuka_base_robot(self):
        bot = Kuka()
        bot.tf_base = pose_x(0.5, 0.1, 0.2, 0.3)
        q_test = self.generate_random_configurations(bot)
        for qi in q_test:
            Tactual = bot.fk(qi)
            Tdesired = fki.fk_kuka(qi)
            Tdesired = bot.tf_base @ Tdesired
            assert_almost_equal(Tactual, Tdesired)

    def test_Kuka_on_rail_robot(self):
        bot = KukaOnRail()
        q_test = self.generate_random_configurations(bot)
        for qi in q_test:
            Tactual = bot.fk(qi)
            Tdesired = fki.fk_kuka(qi[1:])
            Tdesired = pose_x(PI / 2, 0, 0, qi[0]) @ Tdesired
            assert_almost_equal(Tactual, Tdesired)

    def test_PlanarArm_base(self):
        robot1.tf_base = pose_x(1.5, 0.3, 0.5, 1.2)
        q_test = self.generate_random_configurations(robot1)
        for qi in q_test:
            Tactual = robot1.fk(qi)
            Tdesired = fki.fk_PlanarArm(qi, robot1.links)
            Tdesired = robot1.tf_base @ Tdesired
            assert_almost_equal(Tactual, Tdesired)

    def test_estimate_max_extend(self):
        m = robot1.estimate_max_extension()
        assert_almost_equal(m, 3)


class TestPlotKinematics:
    def test_plot_kuka(self):
        robot = Kuka()

        fig = plt.figure()
        ax = fig.gca(projection="3d")
        robot.plot_kinematics(ax, np.ones(robot.ndof))
        # plt.show(block=True)
