from numpy import ndarray
from typing import List
from ..path.sampling import SamplingSetting
from .types import SolveMethod, CostFuntionType


class OptSettings:
    """
    Settings for the numerical optimization based planners.
    """

    def __init__(
        self,
        q_init: ndarray = None,
        max_iters: int = None,
        weights: List[float] = None,
        con_objective_weight=0.0,
    ):
        # q init is handled when whe know the path length and the ndof of the robot
        self.q_init = q_init
        self.weights = weights
        self.con_objective_weight = con_objective_weight
        if max_iters is None:
            self.max_iters = 100
        else:
            self.max_iters = max_iters


class SolverSettings:
    def __init__(
        self,
        solve_method: SolveMethod,
        cost_function_type: CostFuntionType,
        sampling_settings: SamplingSetting = None,
        opt_settings: OptSettings = None,
    ):
        self.solve_method = solve_method
        self.cost_function_type = cost_function_type

        if solve_method == SolveMethod.sampling_based:
            assert sampling_settings is not None
            self.sampling_settings = sampling_settings
            # fill in the correct cost function based on the type

        elif solve_method == SolveMethod.optimization_based:
            assert opt_settings is not None
            self.opt_settings = opt_settings
