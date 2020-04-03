import numpy as np
from typing import List
from abc import ABC, abstractmethod

from acrolib.sampling import Sampler, SampleMethod
from acrolib.quaternion import Quaternion
from acrolib.geometry import (
    rpy_to_rot_mat,
    rotation_matrix_to_rpy,
    tf_inverse,
    quat_distance,
)

from acrobotics.robot import Robot

from .sampling import SamplingSetting, SearchStrategy

from .tolerance import Tolerance, QuaternionTolerance
from .path_pt_base import Pose, PathPt

from .util import (
    create_grid,
    check_rxyz_input,
)


class TolPositionPt(Pose, PathPt):
    def __init__(self, position, quaternion, tolerance: List[Tolerance]):
        # assert len(position) == len(tolerance)
        self.pos = np.array(position)
        self.quat = quaternion
        self.pos_tol = tolerance
        self.tol = tolerance
        self.sampler = Sampler()
        self.sample_dim = super().count_tolerance(tolerance)

    def to_transform(self, sampled_pos):
        tf = np.eye(4)
        tf[:3, :3] = self.rotation_matrix
        tf[:3, 3] = np.array(sampled_pos)
        return tf

    def transform_to_rel_tolerance_deviation(self, tf):
        """ Convert tf in world frame to position expressed in local path frame. """
        T_rel = tf_inverse(self.transformation_matrix) @ tf
        return T_rel[:3, 3]

    def rel_to_abs(self, grid):
        R = self.rotation_matrix
        X_rel, Y_rel, Z_rel = R[:, 0], R[:, 1], R[:, 2]

        samples = []
        for r in grid:
            dx, dy, dz = r[0], r[1], r[2]
            sample = self.pos + dx * X_rel + dy * Y_rel + dz * Z_rel
            samples.append(sample)

        return samples

    def sample_relative_grid(self):
        tol_ranges = [tol.discretize() for tol in self.tol]
        return create_grid(tol_ranges)

    def sample_grid(self):
        relative_grid = self.sample_relative_grid()
        samples = self.rel_to_abs(relative_grid)
        return [self.to_transform(sample) for sample in samples]

    def sample_incremental(self, num_samples, method: SampleMethod):
        R = self.sampler.sample(num_samples, self.sample_dim, method)

        # scale samples from range [0, 1] to desired range
        relative_grid = np.zeros((num_samples, len(self.tol)))
        cnt = 0
        for i, value in enumerate(self.tol):
            if value.has_tolerance:
                relative_grid[:, i] = (
                    R[:, cnt] * (value.upper - value.lower) + value.lower
                )
                cnt += 1
            else:
                relative_grid[:, i] = np.zeros(num_samples)
        samples = self.rel_to_abs(relative_grid)

        return [self.to_transform(sample) for sample in samples]


class TolEulerPt(Pose, PathPt):
    def __init__(
        self, position, quaternion, pos_tol: List[Tolerance], rot_tol: List[Tolerance]
    ):
        self.pos = np.array(position)
        self.quat = quaternion
        self.pos_tol = pos_tol
        self.rot_tol = rot_tol
        self.tol = pos_tol + rot_tol
        self.sample_dim = super().count_tolerance(self.tol)
        self.sampler = Sampler()

    def to_transform(self, sample):
        tf = np.eye(4)
        tf[:3, 3] = sample[:3]
        tf[:3, :3] = self.rotation_matrix @ rpy_to_rot_mat(sample[3:])
        return tf

    def transform_to_rel_tolerance_deviation(self, tf):
        """ Convert tf in world frame to position expressed in local path frame. """
        T_rel = tf_inverse(self.transformation_matrix) @ tf
        rxyz = rotation_matrix_to_rpy(T_rel[:3, :3])
        rxyz = np.array(rxyz)
        return np.hstack((T_rel[:3, 3], rxyz))

    def rel_to_abs(self, grid):
        R = self.rotation_matrix
        X_rel, Y_rel, Z_rel = R[:, 0], R[:, 1], R[:, 2]

        samples = []
        for r in grid:
            dx, dy, dz = r[0], r[1], r[2]
            sample = np.zeros(6)
            sample[:3] = self.pos + dx * X_rel + dy * Y_rel + dz * Z_rel
            sample[3:] = r[3:]
            samples.append(sample)

        return [self.to_transform(sample) for sample in samples]

    def sample_grid(self):
        tol_ranges = [tol.discretize() for tol in self.tol]
        relative_grid = create_grid(tol_ranges)

        return self.rel_to_abs(relative_grid)

    def sample_incremental(self, num_samples, method: SampleMethod):
        R = self.sampler.sample(num_samples, self.sample_dim, method)

        # scale samples from range [0, 1] to desired range
        relative_grid = np.zeros((num_samples, len(self.tol)))
        cnt = 0
        for i, value in enumerate(self.tol):
            if value.has_tolerance:
                relative_grid[:, i] = (
                    R[:, cnt] * (value.upper - value.lower) + value.lower
                )
                cnt += 1
            else:
                relative_grid[:, i] = np.zeros(num_samples)

        return self.rel_to_abs(relative_grid)


class TolQuatPt(Pose, PathPt):
    def __init__(
        self,
        position,
        quaternion,
        pos_tol: List[Tolerance],
        quat_tol: QuaternionTolerance,
    ):
        self.pos = np.array(position)
        self.quat = quaternion
        self.pos_tol = pos_tol
        self.quat_tol = quat_tol
        self.tol = pos_tol + [quat_tol]
        # TODO not using the sampler for the quaternions at the moment
        self.sample_dim = self.count_tolerance(
            self.pos_tol
        )  #  1 uniform and 3 gaussian TODO
        self.sampler = Sampler()

    def to_transform(self, pos_sample, quat_sample):
        tf = np.eye(4)
        tf[:3, 3] = pos_sample
        tf[:3, :3] = quat_sample.rotation_matrix
        return tf

    def transform_to_rel_tolerance_deviation(self, tf):
        """ Convert tf in world frame to position expressed in local path frame. """
        T_rel = tf_inverse(self.transformation_matrix) @ tf
        q = Quaternion(matrix=tf)
        return np.hstack((T_rel[:3, 3], quat_distance(q, self.quat)))

    def rel_to_abs(self, pos_grid):
        R = self.rotation_matrix
        X_rel, Y_rel, Z_rel = R[:, 0], R[:, 1], R[:, 2]

        pos_samples = []
        for r in pos_grid:
            dx, dy, dz = r[0], r[1], r[2]
            sample = self.pos + dx * X_rel + dy * Y_rel + dz * Z_rel
            pos_samples.append(sample)

        return pos_samples

    def sample_grid(self):
        raise NotImplementedError

    def sample_incremental(self, num_samples, method: SampleMethod):

        # sample position if toleranced
        R = self.sampler.sample(num_samples, self.sample_dim, method)

        # scale samples from range [0, 1] to desired range
        relative_grid = np.zeros((num_samples, len(self.pos_tol)))
        cnt = 0
        for i, value in enumerate(self.pos_tol):
            if value.has_tolerance:
                relative_grid[:, i] = (
                    R[:, cnt] * (value.upper - value.lower) + value.lower
                )
                cnt += 1
            else:
                relative_grid[:, i] = np.zeros(num_samples)

        pos_samples = self.rel_to_abs(relative_grid)

        # sample orientation
        if method == SampleMethod.random_uniform:
            quat_samples = [
                self.quat.random_near(self.quat_tol.dist) for _ in range(num_samples)
            ]

        else:
            raise NotImplementedError

        return [self.to_transform(p, q) for p, q in zip(pos_samples, quat_samples)]


class FreeOrientationPt(TolQuatPt):
    """TODO self.quat does not mean necessary mean anything for this type of point,
    but is used in transform_to_rel_tolerance_deviation of the parent class.

    """

    def __init__(self, position, pos_tol: List[Tolerance]):
        super().__init__(
            position, Quaternion(), pos_tol, QuaternionTolerance(0.25 * np.pi + 0.01)
        )
