import numpy as np
from .ik_result import IKResult


def ik(T, tf_base) -> IKResult:
    """ TODO add base frame correction
    """
    Rbase = tf_base[:3, :3]
    Ree = T[:3, :3]
    Ree_rel = np.dot(Rbase.transpose(), Ree)
    # ignore position
    # n s a according to convention Siciliano
    n = Ree_rel[:3, 0]
    s = Ree_rel[:3, 1]
    a = Ree_rel[:3, 2]

    A = np.sqrt(a[0] ** 2 + a[1] ** 2)
    # solution with theta2 in (0, pi)
    t1_1 = np.arctan2(a[1], a[0])
    t2_1 = np.arctan2(A, a[2])
    t3_1 = np.arctan2(s[2], -n[2])
    # solution with theta2 in (-pi, 0)
    t1_2 = np.arctan2(-a[1], -a[0])
    t2_2 = np.arctan2(-A, a[2])
    t3_2 = np.arctan2(-s[2], n[2])

    q_sol = np.zeros((2, 3))
    q_sol[0, 0], q_sol[0, 1], q_sol[0, 2] = t1_1, t2_1, t3_1
    q_sol[1, 0], q_sol[1, 1], q_sol[1, 2] = t1_2, t2_2, t3_2
    return IKResult(True, q_sol)
