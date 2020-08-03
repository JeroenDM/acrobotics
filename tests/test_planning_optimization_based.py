import numpy as np
import acrobotics as ab
import matplotlib.pyplot as plt

from acrolib.quaternion import Quaternion
from acrolib.sampling import SampleMethod
from acrolib.plotting import get_default_axes3d

from acrobotics.robot import Robot


def test_complete_problem():
    path_ori_free = []
    for s in np.linspace(0, 1, 3):
        xi = 0.8
        yi = s * 0.2 + (1 - s) * (-0.2)
        zi = 0.2
        path_ori_free.append(
            ab.FreeOrientationPt(
                [xi, yi, zi], [ab.NoTolerance(), ab.NoTolerance(), ab.NoTolerance()]
            )
        )

    table = ab.Box(0.5, 0.5, 0.1)
    table_tf = np.array(
        [[1, 0, 0, 0.80], [0, 1, 0, 0.00], [0, 0, 1, 0.12], [0, 0, 0, 1]]
    )
    scene1 = ab.Scene([table], [table_tf])

    robot = ab.Kuka()
    # robot.tool = torch

    settings = ab.SamplingSetting(
        ab.SearchStrategy.INCREMENTAL,
        1,
        SampleMethod.random_uniform,
        500,
        tolerance_reduction_factor=2,
    )

    solve_set = ab.SolverSettings(
        ab.SolveMethod.sampling_based,
        ab.CostFuntionType.sum_squared,
        sampling_settings=settings,
    )

    setup = ab.PlanningSetup(robot, path_ori_free, scene1)

    sol1 = ab.solve(setup, solve_set)
    assert sol1.success

    s2 = ab.SolverSettings(
        ab.SolveMethod.optimization_based,
        ab.CostFuntionType.sum_squared,
        opt_settings=ab.OptSettings(q_init=np.array(sol1.joint_positions)),
    )

    sol2 = ab.solve(setup, s2)

    assert sol2.success

    # fig, ax = get_default_axes3d()
    # scene1.plot(ax, c="g")
    # robot.animate_path(fig, ax, sol2.joint_positions)
    # plt.show(block=True)
