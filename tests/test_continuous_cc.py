# -*- coding: utf-8 -*-

"""
Try continous collision checking for a simple path through an obstacle.
"""
import time
import fcl
import numpy as np
import matplotlib.pyplot as plt

from acrolib.plotting import get_default_axes3d, plot_reference_frame
from acrolib.geometry import translation

from acrobotics.robot_examples import Kuka
from acrobotics.tool_examples import torch2
from acrobotics.geometry import Scene
from acrobotics.shapes import Box

robot = Kuka()
robot.tool = torch2

DEBUG = False


def show_animation(robot, scene, qa, qb):
    q_path = np.linspace(qa, qb, 10)
    fig, ax = get_default_axes3d([-0.8, 0.8], [0, 1.6], [-0.2, 1.4])
    ax.set_axis_off()
    ax.view_init(elev=31, azim=-15)
    scene.plot(ax, c="green")
    robot.animate_path(fig, ax, q_path)
    plt.show()


def test_ccd_1():
    table = Box(2, 2, 0.1)
    T_table = translation(0, 0, -0.2)
    obstacle = Box(0.01, 0.01, 1.5)
    T_obs = translation(0, 0.5, 0.55)
    scene = Scene([table, obstacle], [T_table, T_obs])
    q_start = np.array([1.0, 1.5, -0.3, 0, 0, 0])
    q_goal = np.array([2.0, 1.5, 0.3, 0, 0, 0])

    res = robot.is_path_in_collision(q_start, q_goal, scene)
    assert res

    if DEBUG:
        print("resut test 1: ", res)
        show_animation(robot, scene, q_start, q_goal)


def test_ccd_2():
    table = Box(2, 2, 0.1)
    T_table = translation(0, 0, -0.2)
    obstacle = Box(0.2, 0.1, 0.01)
    T_obs = translation(0, 0.9, 0.55)
    scene = Scene([table, obstacle], [T_table, T_obs])
    q_start = np.array([1.5, 1.5, -0.3, 0, 0.3, 0])
    q_goal = np.array([1.5, 1.5, 0.3, 0, -0.3, 0])

    res = robot.is_path_in_collision(q_start, q_goal, scene)
    assert res

    if DEBUG:
        print("resut test 2: ", res)
        show_animation(robot, scene, q_start, q_goal)


def test_ccd_3():
    table = Box(2, 2, 0.1)
    T_table = translation(0, 0, -0.2)
    obstacle = Box(0.01, 0.2, 0.2)
    T_obs = translation(0, 1.2, 0)
    scene = Scene([table, obstacle], [T_table, T_obs])
    q_start = np.array([1.0, 1.2, -0.5, 0, 0, 0])
    q_goal = np.array([2.0, 1.2, -0.5, 0, 0, 0])

    res = robot.is_path_in_collision(q_start, q_goal, scene)
    assert res

    if DEBUG:
        print("resut test 3: ", res)
        show_animation(robot, scene, q_start, q_goal)


if __name__ == "__main__":
    test_ccd_1()
    test_ccd_2()
    test_ccd_3()
