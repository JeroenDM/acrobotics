import numpy as np
from .ik_result import IKResult


def ik(T, links) -> IKResult:
    # ignore orientation
    px, py, pz = T[0, 3], T[1, 3], T[2, 3]
    l2, l3 = links[1].dh.a, links[2].dh.a
    tol = 1e-6
    Lps = px ** 2 + py ** 2 + pz ** 2

    # Theta 3
    # =======
    c3 = (Lps - l2 ** 2 - l3 ** 2) / (2 * l2 * l3)

    if c3 > (1 + tol) or c3 < -(1 + tol):
        return IKResult(False)
    elif c3 > (1 - tol) or c3 < -(1 - tol):
        # almost 1 or -1 => fix it
        c3 = np.sign(c3)

    # point should be reachable
    # TODO just use arccos
    s3 = np.sqrt(1 - c3 ** 2)

    t3_1, t3_2 = np.arctan2(s3, c3), np.arctan2(-s3, c3)

    # Theta 2
    # =======
    # TODO must be greater than zero (numerical error?)
    Lxy = np.sqrt(px ** 2 + py ** 2)
    A = l2 + l3 * c3  # often used expression
    B = l3 * s3
    # positive sign for s3
    t2_1 = np.arctan2(pz * A - Lxy * B, Lxy * A + pz * B)
    t2_2 = np.arctan2(pz * A + Lxy * B, -Lxy * A + pz * B)
    # negative sign for s3
    t2_3 = np.arctan2(pz * A + Lxy * B, Lxy * A - pz * B)
    t2_4 = np.arctan2(pz * A - Lxy * B, -Lxy * A - pz * B)

    # Theta 1
    # =======
    t1_1 = np.arctan2(py, px)
    t1_2 = np.arctan2(-py, -px)

    # 4 solutions
    # =========
    q_sol = [
        [t1_1, t2_1, t3_1],
        [t1_1, t2_3, t3_2],
        [t1_2, t2_2, t3_1],
        [t1_2, t2_4, t3_2],
    ]

    return IKResult(True, np.array(q_sol))

