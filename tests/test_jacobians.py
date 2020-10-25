import numpy as np
import acrobotics as ab

from numpy.testing import assert_almost_equal

NUM_RANDOM_TESTS = 10


def numeric_jacobian(fun, qi):
    fi = fun(qi)
    n_in, n_out = len(qi), len(fi)

    J = np.zeros((n_out, n_in))
    h = 1e-7
    Ih = np.eye(n_in) * h
    for col in range(n_in):
        J[:, col] = (fun(qi + Ih[col]) - fi) / h
    return J


def dummy_f(x):
    """ Simple function to test numerical differentiation. """
    assert len(x) == 2
    o1 = np.sin(x[0])
    o2 = np.cos(x[1])
    o3 = x[0] + x[1] ** 2
    return np.array([o1, o2, o3])


def dummy_f_diff(x):
    """ Jacobian of the dummy_f. """
    J = np.zeros((3, 2))
    J[0, 0] = np.cos(x[0])
    J[1, 0] = 0.0
    J[2, 0] = 1.0

    J[0, 1] = 0.0
    J[1, 1] = -np.sin(x[1])
    J[2, 1] = 2 * x[1]
    return J


def test_numerical_diff():
    for _ in range(NUM_RANDOM_TESTS):
        x = np.random.rand(2)
        j_exact = dummy_f_diff(x)
        j_approx = numeric_jacobian(dummy_f, x)
        assert_almost_equal(j_exact, j_approx)


def test_planer_arm():
    robot = ab.PlanarArm()

    for _ in range(NUM_RANDOM_TESTS):
        q = np.random.rand(robot.ndof)
        j_exact = robot.jacobian_position(q)
        j_approx = numeric_jacobian(lambda q: robot.fk(q)[:3, 3], q)
        assert_almost_equal(j_exact, j_approx)


def test_planer_arm_rpy():
    robot = ab.PlanarArm()

    q = np.random.rand(robot.ndof)
    print(robot.fk_rpy_casadi(q))

    for _ in range(NUM_RANDOM_TESTS):
        q = np.random.rand(robot.ndof)
        j_exact = robot.jacobian_rpy(q)
        j_approx = numeric_jacobian(lambda q: robot.fk_rpy(q), q)
        assert_almost_equal(j_exact, j_approx)


def test_kuka():
    robot = ab.Kuka()

    for _ in range(NUM_RANDOM_TESTS):
        q = np.random.rand(robot.ndof)
        j_exact = robot.jacobian_position(q)
        j_approx = numeric_jacobian(lambda q: robot.fk(q)[:3, 3], q)
        assert_almost_equal(j_exact, j_approx)


def test_kuka_rpy():
    robot = ab.Kuka()

    q = np.random.rand(robot.ndof)
    print(robot.fk_rpy_casadi(q))

    for _ in range(NUM_RANDOM_TESTS):
        q = np.random.rand(robot.ndof)
        j_exact = robot.jacobian_rpy(q)
        j_approx = numeric_jacobian(lambda q: robot.fk_rpy(q), q)
        assert_almost_equal(j_exact[:3, :], j_approx[:3, :])
        assert_almost_equal(j_exact[3:, :], j_approx[3:, :], decimal=5)
