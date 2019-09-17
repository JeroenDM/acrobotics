import numpy as np
import matplotlib.pyplot as plt

from acrolib.quaternion import Quaternion
from acrolib.sampling import SampleMethod

from acrobotics.util import get_default_axes3d
from acrobotics.robot import Robot
from acrobotics.robot_examples import Kuka
from acrobotics.geometry import ShapeSoup
from acrobotics.shapes import Box

from acrobotics.path.tolerance import NoTolerance, QuaternionTolerance
from acrobotics.path.path_pt import FreeOrientationPt
from acrobotics.path.sampling import SamplingSetting, SearchStrategy
from acrobotics.planning.types import (
    CostFuntionType,
    SolveMethod,
    Solution,
    PlanningSetup,
)
from acrobotics.planning.settings import SolverSettings, OptSettings
from acrobotics.planning.solver import solve


def test_complete_problem():
    path_ori_free = []
    for s in np.linspace(0, 1, 3):
        xi = 0.8
        yi = s * 0.2 + (1 - s) * (-0.2)
        zi = 0.2
        path_ori_free.append(
            FreeOrientationPt(
                [xi, yi, zi], [NoTolerance(), NoTolerance(), NoTolerance()]
            )
        )

    table = Box(0.5, 0.5, 0.1)
    table_tf = np.array(
        [[1, 0, 0, 0.80], [0, 1, 0, 0.00], [0, 0, 1, 0.12], [0, 0, 0, 1]]
    )
    scene1 = ShapeSoup([table], [table_tf])

    robot = Kuka()
    # robot.tool = torch

    settings = SamplingSetting(
        SearchStrategy.INCREMENTAL,
        1,
        SampleMethod.random_uniform,
        500,
        tolerance_reduction_factor=2,
    )

    solve_set = SolverSettings(
        SolveMethod.sampling_based,
        CostFuntionType.sum_squared,
        sampling_settings=settings,
    )

    setup = PlanningSetup(robot, path_ori_free, scene1)

    sol1 = solve(setup, solve_set)
    assert sol1.success

    s2 = SolverSettings(
        SolveMethod.optimization_based,
        CostFuntionType.sum_squared,
        opt_settings=OptSettings(q_init=np.array(sol1.joint_positions)),
    )

    sol2 = solve(setup, s2)

    assert sol2.success

    # fig, ax = get_default_axes3d()
    # scene1.plot(ax, c="g")
    # robot.animate_path(fig, ax, sol2.joint_positions)
    # plt.show(block=True)
