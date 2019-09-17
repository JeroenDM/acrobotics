import numpy as np
from .robot import Robot
from .inverse_kinematics.ik_result import IKResult
from tqdm import tqdm

from pyquaternion import Quaternion


class EnvelopeSettings:
    def __init__(
        self,
        sample_distance: float,
        num_orientation_samples: int,
        max_ik_solutions: int,
    ):
        self.sample_distance = sample_distance
        self.num_orientation_samples = num_orientation_samples
        self.max_ik_solutions = max_ik_solutions


def sample_position(position: np.ndarray, n: int):
    """ Return n random transforms at the given position. """
    tf_samples = [Quaternion.random().transformation_matrix for _ in range(n)]
    for tfi in tf_samples:
        tfi[:3, 3] = position
    return tf_samples


def process_ik_solution(robot: Robot, ik_solution: IKResult):
    """ Return the number of collision free ik_solutions. """
    if ik_solution.success:
        q_collision_free = []
        for qi in ik_solution.solutions:
            if not robot.is_in_self_collision(qi):
                q_collision_free.append(qi)
        return len(q_collision_free)
    else:
        return 0


def calculate_reachability(
    robot: Robot,
    position: np.ndarray,
    num_samples: int = 100,
    max_ik_solutions: int = 8,
):
    """
    Return the fraction of reachable poses at a given position by solving
    the inverse kinematics for uniform random orientation samples.
    """
    sampled_transforms = sample_position(position, n=num_samples)
    reachable_cnt = 0
    for transfom in sampled_transforms:
        ik_sol = robot.ik(transfom)
        reachable_cnt += process_ik_solution(robot, ik_sol)
    return reachable_cnt / (max_ik_solutions * num_samples)


def scale(X, from_range, to_range):
    Y = (X - from_range[0]) / (from_range[1] - from_range[0])
    Y = Y * (to_range[1] - to_range[0]) + to_range[0]
    return Y


def generate_positions(max_extension, num_points: int):
    steps = np.linspace(-max_extension, max_extension, num_points)
    x, y, z = np.meshgrid(steps, steps, steps)
    return np.vstack((x.flatten(), y.flatten(), z.flatten())).T
    # return np.stack((x, y, z))


def generate_robot_envelope(robot: Robot, settings: EnvelopeSettings):
    max_extension = robot.estimate_max_extension()
    num_points = int(2 * max_extension / settings.sample_distance)
    points = generate_positions(max_extension, num_points)
    result = np.zeros((len(points), 4))

    for i, point in enumerate(tqdm(points)):
        reachability = calculate_reachability(
            robot, point, settings.num_orientation_samples, settings.max_ik_solutions
        )  # TODO: other params
        result[i, 0] = reachability
        result[i, 1:] = point

    return result
