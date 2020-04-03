import pytest
import numpy as np
from numpy.testing import assert_almost_equal

# import matplotlib.pyplot as plt

from acrolib.quaternion import Quaternion
from acrolib.sampling import SampleMethod
from acrolib.plotting import get_default_axes3d, plot_reference_frame

from acrobotics.shapes import Box
from acrobotics.geometry import Scene
from acrobotics.robot import Robot
from acrobotics.robot_examples import Kuka
from acrobotics.path.sampling import SamplingSetting, SearchStrategy
from acrobotics.path.tolerance import (
    NoTolerance,
    Tolerance,
    QuaternionTolerance,
    SymmetricTolerance,
)
from acrobotics.path.factory import create_line, create_arc
from acrobotics.path.path_pt import TolPositionPt, TolEulerPt, TolQuatPt
from acrobotics.planning.solver import solve
from acrobotics.planning.types import SolveMethod, CostFuntionType, PlanningSetup
from acrobotics.planning.settings import SolverSettings


class DummyRobot(Robot):
    def __init__(self, is_colliding=False):
        self.is_colliding = is_colliding
        self.ndof = 6

    def is_in_collision(self, joint_position, scene=None):
        return self.is_colliding

    def fk(self, q):
        tf = np.eye(4)
        tf[:3, 3] = np.array([1, -2.09, 3])
        return tf


def test_tol_quat_pt_with_weights():
    path_ori_free = []
    for s in np.linspace(0, 1, 3):
        xi = 0.8
        yi = s * 0.2 + (1 - s) * (-0.2)
        zi = 0.2
        path_ori_free.append(
            TolQuatPt(
                [xi, yi, zi],
                Quaternion(axis=[1, 0, 0], angle=np.pi),
                [NoTolerance(), NoTolerance(), NoTolerance()],
                QuaternionTolerance(2.0),
            )
        )

    table = Box(0.5, 0.5, 0.1)
    table_tf = np.array(
        [[1, 0, 0, 0.80], [0, 1, 0, 0.00], [0, 0, 1, 0.12], [0, 0, 0, 1]]
    )
    scene1 = Scene([table], [table_tf])

    robot = Kuka()
    # robot.tool = torch
    setup = PlanningSetup(robot, path_ori_free, scene1)

    # weights to express the importance of the joints in the cost function
    joint_weights = [10.0, 5.0, 1.0, 1.0, 1.0, 1.0]

    settings = SamplingSetting(
        SearchStrategy.INCREMENTAL,
        sample_method=SampleMethod.random_uniform,
        num_samples=500,
        iterations=2,
        tolerance_reduction_factor=2,
        weights=joint_weights,
    )

    solve_set = SolverSettings(
        SolveMethod.sampling_based,
        CostFuntionType.weighted_sum_squared,
        sampling_settings=settings,
    )

    sol = solve(setup, solve_set)
    assert sol.success

    for qi, s in zip(sol.joint_positions, np.linspace(0, 1, 3)):
        xi = 0.8
        yi = s * 0.2 + (1 - s) * (-0.2)
        zi = 0.2

        fk = robot.fk(qi)
        pos_fk = fk[:3, 3]
        assert_almost_equal(pos_fk, np.array([xi, yi, zi]))

    # fig, ax = get_default_axes3d()
    # scene1.plot(ax, c="g")
    # robot.animate_path(fig, ax, sol.joint_positions)
    # plt.show(block=True)


def test_tol_position_pt_planning_problem():
    robot = Kuka()

    table = Box(0.5, 0.5, 0.1)
    table_tf = np.array(
        [[1, 0, 0, 0.80], [0, 1, 0, 0.00], [0, 0, 1, 0.12], [0, 0, 0, 1]]
    )
    scene1 = Scene([table], [table_tf])

    # create path
    quat = Quaternion(axis=np.array([1, 0, 0]), angle=np.pi)
    tolerance = [NoTolerance(), SymmetricTolerance(0.05, 10), NoTolerance()]
    first_point = TolPositionPt(np.array([0.9, -0.2, 0.2]), quat, tolerance)
    # end_position = np.array([0.9, 0.2, 0.2])

    # path = create_line(first_point, end_position, 5)
    path = create_arc(
        first_point, np.array([0.9, 0.0, 0.2]), np.array([0, 0, 1]), 2 * np.pi, 5
    )

    planner_settings = SamplingSetting(SearchStrategy.GRID, iterations=1)

    solver_settings = SolverSettings(
        SolveMethod.sampling_based,
        CostFuntionType.sum_squared,
        sampling_settings=planner_settings,
    )

    setup = PlanningSetup(robot, path, scene1)

    sol = solve(setup, solver_settings)
    assert sol.success

    for qi, pt in zip(sol.joint_positions, path):
        fk = robot.fk(qi)
        pos_fk = fk[:3, 3]
        pos_pt = pt.pos
        R_pt = pt.rotation_matrix
        pos_error = R_pt.T @ (pos_fk - pos_pt)
        assert_almost_equal(pos_error[0], 0)
        assert_almost_equal(pos_error[2], 0)
        assert pos_error[1] <= (0.05 + 1e-6)
        assert pos_error[1] >= -(0.05 + 1e-6)


# TODO fix this test
def test_euler_pt_planning_problem():
    robot = Kuka()

    table = Box(0.5, 0.5, 0.1)
    table_tf = np.array(
        [[1, 0, 0, 0.80], [0, 1, 0, 0.00], [0, 0, 1, 0.00], [0, 0, 0, 1]]
    )
    scene1 = Scene([table], [table_tf])

    # create path
    quat = Quaternion(axis=np.array([1, 0, 0]), angle=-3 * np.pi / 4)

    pos_tol = 3 * [NoTolerance()]
    # rot_tol = 3 * [NoTolerance()]
    rot_tol = [
        NoTolerance(),
        SymmetricTolerance(np.pi / 4, 20),
        SymmetricTolerance(np.pi, 20),
    ]
    first_point = TolEulerPt(np.array([0.9, -0.1, 0.2]), quat, pos_tol, rot_tol)
    # end_position = np.array([0.9, 0.1, 0.2])

    # path = create_line(first_point, end_position, 5)
    path = create_arc(
        first_point, np.array([0.9, 0.0, 0.2]), np.array([0, 0, 1]), 2 * np.pi, 5
    )

    planner_settings = SamplingSetting(
        SearchStrategy.GRID, iterations=1, tolerance_reduction_factor=2
    )

    solver_settings = SolverSettings(
        SolveMethod.sampling_based,
        CostFuntionType.sum_squared,
        sampling_settings=planner_settings,
    )

    setup = PlanningSetup(robot, path, scene1)

    sol = solve(setup, solver_settings)
    assert sol.success

    # fig, ax = get_default_axes3d()
    # scene1.plot(ax, c="g")

    # path_tf = [pt.transformation_matrix for pt in path]
    # for tf in path_tf:
    #     plot_reference_frame(ax, tf, 0.1)

    # # for tf in path[0].sample_grid():
    # #     plot_reference_frame(ax, tf, 0.1)

    # fk_tfs = [robot.fk(qi) for qi in sol.joint_positions]
    # for tf in fk_tfs:
    #     plot_reference_frame(ax, tf, 0.1)

    # ax.set_axis_off()
    # robot.animate_path(fig, ax, sol.joint_positions)
    # plt.show(block=True)


def test_state_cost():
    robot = Kuka()

    table = Box(0.5, 0.5, 0.1)
    table_tf = np.array(
        [[1, 0, 0, 0.80], [0, 1, 0, 0.00], [0, 0, 1, 0.00], [0, 0, 0, 1]]
    )
    scene1 = Scene([table], [table_tf])

    # create path
    quat = Quaternion(axis=np.array([1, 0, 0]), angle=-3 * np.pi / 4)

    pos_tol = 3 * [NoTolerance()]
    # rot_tol = 3 * [NoTolerance()]
    rot_tol = [
        NoTolerance(),
        SymmetricTolerance(np.pi / 4, 20),
        SymmetricTolerance(np.pi, 20),
    ]
    first_point = TolEulerPt(np.array([0.9, -0.1, 0.2]), quat, pos_tol, rot_tol)
    # end_position = np.array([0.9, 0.1, 0.2])

    # path = create_line(first_point, end_position, 5)
    path = create_arc(
        first_point, np.array([0.9, 0.0, 0.2]), np.array([0, 0, 1]), 2 * np.pi, 5
    )

    planner_settings = SamplingSetting(
        SearchStrategy.GRID,
        iterations=1,
        tolerance_reduction_factor=2,
        use_state_cost=True,
        state_cost_weight=10.0,
    )

    solver_settings = SolverSettings(
        SolveMethod.sampling_based,
        CostFuntionType.sum_squared,
        sampling_settings=planner_settings,
    )

    setup = PlanningSetup(robot, path, scene1)

    sol = solve(setup, solver_settings)
    assert sol.success


def test_exceptions():
    settings = SamplingSetting(
        SearchStrategy.INCREMENTAL,
        sample_method=SampleMethod.random_uniform,
        num_samples=500,
        iterations=2,
        tolerance_reduction_factor=2,
    )

    solve_set = SolverSettings(
        SolveMethod.sampling_based,
        CostFuntionType.weighted_sum_squared,
        sampling_settings=settings,
    )

    setup = PlanningSetup(None, None, None)
    with pytest.raises(Exception) as e:
        solve(setup, solve_set)
    assert (
        str(e.value)
        == "No weights specified in SamplingSettings for the weighted cost function."
    )

    robot = Kuka()
    scene = Scene([], [])

    pos = np.array([1000, 0, 0])
    quat = Quaternion(axis=np.array([1, 0, 0]), angle=-3 * np.pi / 4)
    path = [TolPositionPt(pos, quat, 3 * [NoTolerance()])]

    solve_set2 = SolverSettings(
        SolveMethod.sampling_based,
        CostFuntionType.sum_squared,
        sampling_settings=settings,
    )

    setup2 = PlanningSetup(robot, path, scene)
    with pytest.raises(Exception) as e:
        solve(setup2, solve_set2)
    assert str(e.value) == f"No valid joint solutions for path point {0}."
