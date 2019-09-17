import time
import numpy as np
from copy import deepcopy
from typing import List

from acrolib.dynamic_programming import shortest_path, shortest_path_with_state_cost
from acrolib.cost_functions import (  # pylint: disable=no-name-in-module
    norm_l1,
    norm_l2,
    sum_squared,
    weighted_sum_squared,
)
from acrolib.quaternion import Quaternion

from ..path.path_pt_base import PathPt
from ..path.path_pt import TolPositionPt, TolEulerPt, TolQuatPt
from ..path.sampling import SamplingSetting
from .types import JointPath, SolveMethod, CostFuntionType, PlanningSetup, Solution
from .settings import SolverSettings


def sampling_based_solve(planning_setup: PlanningSetup, settings: SolverSettings):
    assert settings.solve_method == SolveMethod.sampling_based

    s = settings.sampling_settings

    # if the user has not given a cost function
    if s.cost_function is None:
        if settings.cost_function_type == CostFuntionType.l1_norm:
            s.cost_function = norm_l1
        elif settings.cost_function_type == CostFuntionType.l2_norm:
            s.cost_function = norm_l2
        elif settings.cost_function_type == CostFuntionType.sum_squared:
            s.cost_function = sum_squared
        elif settings.cost_function_type == CostFuntionType.weighted_sum_squared:
            s.cost_function = weighted_sum_squared
            # input checking
            if s.weights is None:
                raise Exception(
                    "No weights specified in SamplingSettings for the weighted cost function."
                )

            # make sure the weights have the right type for the cython functions
            s.weights = np.array(s.weights, dtype="float64")

    # set initial values before starting the iterations
    current_path = [deepcopy(pt) for pt in planning_setup.path]
    current_joint_path = JointPath(None, None)

    # save intermediate results
    all_paths = []
    all_joint_paths = []

    start_time = time.time()
    for step in range(s.iterations):
        all_paths.append(deepcopy(current_path))

        current_path, current_joint_path = solver_step(
            step,
            current_path,
            current_joint_path,
            planning_setup,
            settings.sampling_settings,
        )

        all_joint_paths.append(deepcopy(current_joint_path))

    stop_time = time.time()

    extra_info = {"all_joint_paths": all_joint_paths, "all_paths": all_paths}
    return Solution(
        True,
        current_joint_path.joint_positions,
        current_joint_path.cost,
        stop_time - start_time,
        extra_info,
    )


def solver_step(step_count: int, prev_path, prev_sol, setup, sampling_settings):
    if step_count == 0:
        current_path = prev_path
    else:
        current_path = create_reduced_tolerance_path(
            prev_path, prev_sol.joint_positions, setup.robot
        )

    Q = path_to_joint_solutions(current_path, setup, sampling_settings)
    q_path = find_shortest_joint_path(Q, setup, sampling_settings)

    return current_path, q_path


def path_to_joint_solutions(current_path, setup, sampling_settings):
    JS = []

    for i, pt in enumerate(current_path):
        js = pt.to_joint_solutions(setup.robot, sampling_settings, setup.scene)

        if len(js) == 0:
            raise Exception(f"No valid joint solutions for path point {i}.")

        print(f"Found {len(js)} joint solutions for PathPt {i}")
        JS.append(js)

    return JS


def find_shortest_joint_path(
    joint_solutions: List[np.ndarray], setup, sampling_settings
):
    if sampling_settings.use_state_cost:
        state_cost = calc_state_cost(
            joint_solutions,
            setup.robot,
            setup.path,
            sampling_settings.state_cost_weight,
        )

        if sampling_settings.weights is not None:
            w = sampling_settings.weights
            res = shortest_path_with_state_cost(
                joint_solutions,
                state_cost,
                lambda x, y: sampling_settings.cost_function(x, y, w),
            )
        else:
            res = shortest_path_with_state_cost(
                joint_solutions, state_cost, sampling_settings.cost_function
            )
    else:
        if sampling_settings.weights is not None:
            w = sampling_settings.weights
            res = shortest_path(
                joint_solutions, lambda x, y: sampling_settings.cost_function(x, y, w)
            )
        else:
            res = shortest_path(joint_solutions, sampling_settings.cost_function)

    if not res["success"]:
        raise ValueError("Failed to find a shortest path in joint_solutions.")
    return JointPath(res["path"], res["length"])


def create_reduced_tolerance_path(current_path, current_joint_positions, robot):
    new_path = []
    for q, pt in zip(current_joint_positions, current_path):
        T_fk = robot.fk(q)
        new_pt = create_new_pt(T_fk, q, pt)
        new_path.append(new_pt)

    return new_path


def create_new_pt(
    fk_transform: np.ndarray,
    q: np.ndarray,
    prev_pt: PathPt,
    tolerance_reduction_factor=2.0,
):
    if isinstance(prev_pt, TolQuatPt):
        new_pos = fk_transform[:3, 3]
        new_quat = Quaternion(matrix=fk_transform)
        new_tol = deepcopy(prev_pt.tol)
        ref_values = prev_pt.transform_to_rel_tolerance_deviation(fk_transform)
        for tolerance, ref in zip(new_tol, ref_values):
            tolerance.reduce_tolerance(tolerance_reduction_factor, ref)

        return TolQuatPt(new_pos, new_quat, new_tol[:3], new_tol[3])

    elif isinstance(prev_pt, TolEulerPt):
        new_pos = fk_transform[:3, 3]
        new_quat = Quaternion(matrix=fk_transform)
        new_tol = deepcopy(prev_pt.tol)
        ref_values = prev_pt.transform_to_rel_tolerance_deviation(fk_transform)
        for tolerance, ref in zip(new_tol, ref_values):
            tolerance.reduce_tolerance(tolerance_reduction_factor, ref)

        return TolEulerPt(new_pos, new_quat, new_tol[:3], new_tol[3:])
    else:
        raise NotImplementedError()


def calc_state_cost(joint_solutions, robot, initial_path, weight):
    state_cost = []

    if not isinstance(initial_path[0], TolEulerPt):
        raise Exception("Current state cost only implemented for euler points.")

    for pt, qs in zip(initial_path, joint_solutions):
        fk_transforms = [robot.fk(qi) for qi in qs]

        tol_devs = [pt.transform_to_rel_tolerance_deviation(tf) for tf in fk_transforms]
        tol_devs = np.array(tol_devs)

        # welding cost rx**2 + ry**2
        cost = tol_devs[:, 3] ** 2 + tol_devs[:, 4] ** 2
        assert cost.shape == (len(tol_devs),)
        state_cost.append(weight * cost)

    return state_cost
