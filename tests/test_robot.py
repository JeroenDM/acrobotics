import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from numpy.testing import assert_almost_equal
from acrolib.geometry import pose_x
from acrobotics.robot import Robot
from acrobotics.robot_examples import Kuka
from acrobotics.geometry import ShapeSoup
from acrobotics.shapes import Box
from .fk_implementations import FKImplementations as fki


class TestCollisionChecking:
    def test_kuka_self_collision(self):
        bot = Kuka()
        gl = [l.geometry for l in bot.links]
        q0 = [0, np.pi / 2, 0, 0, 0, 0]
        q_self_collision = [0, 1.5, -1.3, 0, -1.5, 0]
        tf1 = bot.fk_all_links(q0)
        a1 = bot._check_self_collision(tf1, gl)
        assert a1 is False
        tf2 = bot.fk_all_links(q_self_collision)
        a2 = bot._check_self_collision(tf2, gl)
        assert a2 is True

    def test_kuka_collision(self):
        bot = Kuka()
        q0 = [0, np.pi / 2, 0, 0, 0, 0]
        obj1 = ShapeSoup(
            [Box(0.2, 0.3, 0.5), Box(0.1, 0.3, 0.1)],
            [pose_x(0, 0.75, 0, 0.5), pose_x(0, 0.75, 0.5, 0.5)],
        )
        obj2 = ShapeSoup(
            [Box(0.2, 0.3, 0.5), Box(0.1, 0.3, 0.1)],
            [pose_x(0, 0.3, -0.7, 0.5), pose_x(0, 0.75, 0.5, 0.5)],
        )
        a1 = bot.is_in_collision(q0, obj1)
        assert a1 is True
        a2 = bot.is_in_collision(q0, obj2)
        assert a2 is False
