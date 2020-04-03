import numpy as np
from typing import List
from abc import ABC, abstractmethod

from acrolib.sampling import Sampler, SampleMethod
from acrolib.quaternion import Quaternion

from acrobotics.robot import Robot
from acrobotics.geometry import Scene

from .sampling import SamplingSetting, SearchStrategy
from .tolerance import Tolerance, NoTolerance, SymmetricTolerance, QuaternionTolerance


class Pose(ABC):
    """
    All PathPt have a reference pose that we can translate or rotate using
    the interface defined in this abstract base class.
    """

    pos: np.ndarray
    quat: Quaternion

    @property
    def rotation_matrix(self):
        return self.quat.rotation_matrix

    @property
    def transformation_matrix(self):
        tf = self.quat.transformation_matrix
        tf[:3, 3] = self.pos
        return tf

    def translate(self, trans_vec):
        self.pos = self.pos + trans_vec

    def rotate(self, R):
        R_new = R @ self.quat.rotation_matrix
        self.quat = Quaternion(matrix=R_new)


class PathPt(ABC):
    """ A path point specifies end-effector constraints, using tolerance on position or orientation.

    This base class defines the interface that different type of constraints have in common.
    It also implements calculating all joint positions that satisfy this constraints.
    """

    tol: List[Tolerance]

    @abstractmethod
    def sample_grid(self) -> List[np.ndarray]:
        pass

    @abstractmethod
    def sample_incremental(self, num_samples, method: SampleMethod) -> List[np.ndarray]:
        pass

    @abstractmethod
    def to_transform(self, sampled_values) -> np.array:
        pass

    @abstractmethod
    def transform_to_rel_tolerance_deviation(self, tf: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def rel_to_abs(self, grid: np.ndarray):
        pass

    @staticmethod
    def count_tolerance(values):
        return sum([not isinstance(v, NoTolerance) for v in values])

    @staticmethod
    def _calc_ik(robot, samples) -> List:
        """ Calculate all ik solutions for the discrete representation of the region
        within the constraints.
        """
        joint_solutions = []
        for transform in samples:
            ik_result = robot.ik(transform)
            if ik_result.success:
                joint_solutions.extend(ik_result.solutions)
        return joint_solutions

    def to_joint_solutions(
        self, robot: Robot, settings: SamplingSetting, scene: Scene = None
    ) -> np.ndarray:
        """ Return all joint positions that satisfy the constraints."""

        if settings.search_strategy == SearchStrategy.GRID:
            samples = self.sample_grid()
            joint_solutions = self._calc_ik(robot, samples)

        elif settings.search_strategy == SearchStrategy.INCREMENTAL:
            samples = self.sample_incremental(
                settings.num_samples, settings.sample_method
            )
            joint_solutions = self._calc_ik(robot, samples)

        elif settings.search_strategy == SearchStrategy.MIN_INCREMENTAL:
            return self._incremental_search(robot, scene, settings)
        else:
            raise NotImplementedError

        collision_free_js = [
            q for q in joint_solutions if not robot.is_in_collision(q, scene)
        ]

        return np.array(collision_free_js)

    def _incremental_search(self, robot: Robot, scene: Scene, s: SamplingSetting):
        joint_solutions = []
        for _ in range(s.max_search_iters):

            samples = self.sample_incremental(s.step_size, s.sample_method)

            collision_free_js = [
                q
                for q in self._calc_ik(robot, samples)
                if not robot.is_in_collision(q, scene)
            ]
            joint_solutions.extend(collision_free_js)

            if len(joint_solutions) >= s.desired_num_samples:
                return np.array(joint_solutions)

        raise Exception("Maximum iterations reached in to_joint_solutions.")
