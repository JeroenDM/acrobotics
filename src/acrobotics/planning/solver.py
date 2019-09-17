import enum

from typing import List

from ..robot import Robot
from ..geometry import Scene
from ..path.path_pt import PathPt
from .types import JointPath, SolveMethod, PlanningSetup, Solution
from .settings import SolverSettings
from .sampling_based import sampling_based_solve
from .optimization_based import opt_based_solve


def solve(planning_setup, solver_settings) -> Solution:
    if solver_settings.solve_method == SolveMethod.sampling_based:
        return sampling_based_solve(planning_setup, solver_settings)
    elif solver_settings.solve_method == SolveMethod.optimization_based:
        return opt_based_solve(planning_setup, solver_settings)
    else:
        raise Exception(f"Solver method {solver_settings.solve_method} not implemented")
