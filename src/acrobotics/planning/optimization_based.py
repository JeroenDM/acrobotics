import numpy as np
import casadi as ca
from casadi import Opti, dot

from acrobotics.shapes import Polyhedron
from acrobotics.robot import Robot
from acrobotics.geometry import Scene

from ..path.path_pt import FreeOrientationPt, TolPositionPt, TolEulerPt
from .types import SolveMethod, Solution
from .settings import SolverSettings, OptSettings


def opt_based_solve(setup, settings: SolverSettings):
    assert settings.solve_method == SolveMethod.optimization_based
    s = settings.opt_settings
    if s.q_init is None:
        s.q_init = np.zeros((len(setup.path), setup.robot.ndof))
    return get_optimal_path(
        setup.path,
        setup.robot,
        scene=setup.scene,
        q_init=s.q_init,
        max_iters=s.max_iters,
        w=s.weights,
        cow=s.con_objective_weight,
    )


def get_optimal_path(
    path, robot, scene=None, q_init=None, max_iters=100, w=None, cow=0.0
):
    N = len(path)
    if q_init is None:
        q_init = np.zeros((N, robot.ndof))

    if w is None:
        w = np.ones(robot.ndof)
    print("Using weights:")
    print(w)

    opti = ca.Opti()
    q = opti.variable(N, 6)  #  joint variables along path

    # collision constraints
    if scene is not None:
        cons = create_cc(opti, robot, scene, q)
        opti.subject_to(cons)

    # path constraints
    opti.subject_to(create_path_constraints(q, robot, path))

    # objective
    # V = ca.sum1(
    #     ca.sum2((q[:-1, :] - q[1:, :]) ** 2)
    # )  # + 0.05* ca.sumsqr(q) #+ 1 / ca.sum1(q[:, 4]**2)
    V = 0
    for i in range(1, N):
        for k in range(robot.ndof):
            V += w[k] * (q[i, k] - q[i - 1, k]) ** 2

    if cow > 0:
        print(f"Adding path constraints objective term with lambda: {cow}")
        V += cow * create_path_objective(q, robot, path)
    opti.minimize(V)

    p_opts = {}  # casadi options
    s_opts = {"max_iter": max_iters}  # solver options
    opti.solver("ipopt", p_opts, s_opts)
    opti.set_initial(q, q_init)  # 2 3 4 5  converges

    try:
        sol = opti.solve()
    except RuntimeError as e:
        print(e)
        # return opti object to access debug info
        return Solution(False, extra_info={"opti": opti})

    if sol.stats()["success"]:
        return Solution(True, sol.value(q), sol.value(V))
    else:
        return Solution(False)


def create_path_objective(q, robot, path):
    J = 0
    for i in range(1, len(path)):
        T_ee = robot.fk_casadi(q[i, :])
        R_i = path[i].rotation_matrix
        R_i_ee = R_i.T @ T_ee[:3, :3]
        rxyz = rotation_matrix_to_rxyz_casadi(R_i_ee)
        J += rxyz[0] ** 2 + rxyz[1] ** 2
    return J


def create_path_constraints(q, robot, path):
    # assume all path points are of the same type

    if isinstance(path[0], FreeOrientationPt):
        return _free_orientation_cc(q, robot, path)
    elif isinstance(path[0], TolPositionPt):
        return _tol_position_cc(q, robot, path)
    elif isinstance(path[0], TolEulerPt):
        cons = _tol_position_cc(q, robot, path)
        cons.extend(_tol_rxyz_cc(q, robot, path))
        return cons
    else:
        raise NotImplementedError


def _free_orientation_cc(q, robot, path):
    path_cons = []
    xyz = [tp.pos for tp in path]
    for i in range(len(path)):
        # Ti = fk_kuka2(q[i, :])
        Ti = robot.fk_casadi(q[i, :])
        path_cons.append(xyz[i][0] == Ti[0, 3])
        path_cons.append(xyz[i][1] == Ti[1, 3])
        path_cons.append(xyz[i][2] == Ti[2, 3])
    return path_cons


def _tol_position_cc(q, robot, path):
    path_cons = []

    for i, pt in enumerate(path):
        Tr = pt.transformation_matrix
        Ti = robot.fk_casadi(q[i, :])

        T_error = tf_inverse(Tr) @ Ti
        pos_error = T_error[:3, 3]

        # pos_error = pt.rotation_matrix.T @ (Ti[:3, 3] - T_ref[:3, 3])

        for k in range(3):
            if pt.pos_tol[k].has_tolerance:
                path_cons.append(pos_error[k] <= pt.pos_tol[k].upper)
                path_cons.append(pos_error[k] >= pt.pos_tol[k].lower)
            else:
                path_cons.append(pos_error[k] == 0)

    return path_cons


def rotation_matrix_to_rxyz_casadi(R):
    r11, r12, r13 = R[0, 0], R[0, 1], R[0, 1]
    r23 = R[1, 2]
    r33 = R[2, 2]

    r_x = ca.atan2(-r23, r33)
    r_y = ca.atan2(r13, np.sqrt(r11 ** 2 + r12 ** 2))
    r_z = ca.atan2(-r12, r11)

    return [r_x, r_y, r_z]


def _tol_rxyz_cc(q, robot, path):
    path_cons = []
    for i, pt in enumerate(path):
        Tr = pt.transformation_matrix
        Ti = robot.fk_casadi(q[i, :])

        T_error = tf_inverse(Tr) @ Ti
        rxyz_error = rotation_matrix_to_rxyz_casadi(T_error[:3, :3])

        for k in range(3):
            if pt.rot_tol[k].has_tolerance:
                path_cons.append(rxyz_error[k] <= pt.rot_tol[k].upper)
                path_cons.append(rxyz_error[k] >= pt.rot_tol[k].lower)
            else:
                path_cons.append(rxyz_error[k] == 0)

    return path_cons


def tf_inverse(T):
    """ Efficient inverse of a homogenous transform.

    (Normal matrix inversion would be a bad idea.)
    Returns a copy, not inplace!
    """
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].transpose()
    Ti[:3, 3] = np.dot(-Ti[:3, :3], T[:3, 3])
    return Ti


def create_collision_constraints(lam, mu, Ar, Ao, br, bo, eps=1e-6):
    cons = []
    cons.append(-dot(br, lam) - dot(bo, mu) >= eps)
    cons.append(Ar.T @ lam + Ao.T @ mu == 0.0)
    cons.append(dot(Ar.T @ lam, Ar.T @ lam) <= 1.0)
    cons.append(lam >= 0.0)
    cons.append(mu >= 0.0)
    return cons


def create_cc_for_joint_pose(robot, poly_robot, poly_scene, q, lamk, muk):
    """"
    NOTE: assumes only one shape / robot link
    """
    cons = []
    fk = robot.fk_all_links_casadi(q)
    for i in range(robot.ndof):
        Ri = fk[i][:3, :3]
        pi = fk[i][:3, 3]
        for j in range(len(poly_scene)):
            Ar = poly_robot[i].A @ Ri.T
            br = poly_robot[i].b + Ar @ pi
            cons.extend(
                create_collision_constraints(
                    lamk[i][j], muk[i][j], Ar, poly_scene[j].A, br, poly_scene[j].b
                )
            )
    return cons


def create_cc(opti, robot: Robot, scene: Scene, q):

    poly_robot = []
    for link in robot.links:
        poly_robot.extend(link.geometry.get_polyhedrons())

    if robot.geometry_tool is not None:
        poly_robot.extend(robot.geometry_tool.get_polyhedrons())

    poly_scene = scene.get_polyhedrons()

    S = len(poly_robot[0].b)
    nobs = len(poly_scene)
    N, _ = q.shape
    robs = len(poly_robot)
    # dual variables arranged in convenient lists to acces with indices
    lam = [
        [[opti.variable(S) for j in range(nobs)] for i in range(robs)] for k in range(N)
    ]
    mu = [
        [[opti.variable(S) for j in range(nobs)] for i in range(robs)] for k in range(N)
    ]

    cons = []
    for k in range(N):
        cons.extend(
            create_cc_for_joint_pose(
                robot, poly_robot, poly_scene, q[k, :], lam[k], mu[k]
            )
        )

    return cons

