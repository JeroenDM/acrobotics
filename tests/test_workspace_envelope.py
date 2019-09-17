import numpy as np
from numpy.testing import assert_almost_equal
from acrobotics.workspace_envelope import (
    sample_position,
    process_ik_solution,
    calculate_reachability,
    generate_positions,
    generate_robot_envelope,
    EnvelopeSettings,
)
from acrobotics.inverse_kinematics.ik_result import IKResult
from acrobotics.robot import Robot
from acrobotics.robot_examples import Kuka


def test_sample_position():
    pos = np.array([0.1, 0.2, 0.3])
    samples = sample_position(pos, 5)
    for tf in samples:
        assert_almost_equal(tf[:3, 3], pos)

    for i in range(1, len(samples)):
        assert np.any(np.not_equal(samples[i], samples[i - 1]))


class DummyRobot(Robot):
    def __init__(self):
        pass

    def is_in_self_collision(self, q):
        if q[0] < 0.5:
            return False
        else:
            return True


def test_process_ik_result():
    ik_result = IKResult(True, [[0, 0], [0, 0]])
    robot = DummyRobot()
    res = process_ik_solution(robot, ik_result)
    assert res == 2

    ik_result = IKResult(True, [[0, 0], [1, 1]])
    robot = DummyRobot()
    res = process_ik_solution(robot, ik_result)
    assert res == 1

    ik_result = IKResult(True, [[1, 1], [1, 1]])
    robot = DummyRobot()
    res = process_ik_solution(robot, ik_result)
    assert res == 0


def test_generate_envelop():
    robot = Kuka()
    settings = EnvelopeSettings(1.0, 10, 8)
    we = generate_robot_envelope(robot, settings)

    max_extension = robot.estimate_max_extension()
    num_points = int(2 * max_extension / settings.sample_distance)

    assert we.shape == (num_points ** 3, 4)
