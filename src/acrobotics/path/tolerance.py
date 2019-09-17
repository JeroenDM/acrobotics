import numpy as np


class Tolerance:
    """
    A range with upper and lower bound that can be discretized.
    """

    def __init__(self, lower: float, upper: float, num_samples: int):
        self.lower = lower
        self.upper = upper
        self.num_samples = num_samples
        self.has_tolerance = True

    def discretize(self) -> np.ndarray:
        return np.linspace(self.lower, self.upper, self.num_samples)

    def reduce_tolerance(self, reduction_factor: float, reference_value: float):
        max_lower = min(self.lower - reference_value, 0)  # must be negative
        max_upper = max(self.upper - reference_value, 0)  # must be positive
        self.lower = max(self.lower / reduction_factor, max_lower)
        self.upper = min(self.upper / reduction_factor, max_upper)


class NoTolerance(Tolerance):
    def __init__(self):
        super().__init__(0, 0, 1)
        self.has_tolerance = False

    def discretize(self) -> np.ndarray:
        return np.array([0])

    def reduce_tolerance(self, reduction_factor: float, reference_value: float):
        pass


class SymmetricTolerance(Tolerance):
    def __init__(self, dist: float, num_samples: int):
        super().__init__(-dist, dist, num_samples)


class QuaternionTolerance(Tolerance):
    def __init__(self, quat_distance: float):
        self.dist = quat_distance
        self.has_tolerance = True

    def discretize(self):
        """ Future work, implement fixed grid on SO(3). """
        raise NotImplementedError

    def reduce_tolerance(self, reduction_factor: float, reference_value: float):
        self.dist = self.dist / reduction_factor
