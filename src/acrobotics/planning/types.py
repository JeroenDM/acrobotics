import enum


class JointPath:
    def __init__(self, joint_positions, cost):
        self.joint_positions = joint_positions
        self.cost = cost


class SolveMethod(enum.Enum):
    """ Acrobotics.planning implements two types of algorithms. """

    sampling_based = 0
    optimization_based = 1


class CostFuntionType(enum.Enum):
    l1_norm = 0
    l2_norm = 1
    sum_squared = 3
    weighted_sum_squared = 4


class PlanningSetup:
    def __init__(self, robot, path, scene):
        self.robot = robot
        self.path = path
        self.scene = scene


class Solution:
    def __init__(
        self,
        success: bool,
        joint_positions=None,
        path_cost: float = None,
        run_time: float = None,
        extra_info: dict = None,
    ):
        self.success = success
        self.joint_positions = joint_positions
        self.path_cost = path_cost
        self.run_time = run_time
        self.extra_info = extra_info
