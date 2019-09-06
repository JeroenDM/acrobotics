import numpy as np
from .ik_result import IKResult


def ik(T, links) -> IKResult:
    # compensate for tool frame
    #    if self.tf_tool is not None:
    #        print('Arm2: adapt for tool')
    #        print(T)
    #        Tw = np.dot(T, tf_inverse(self.tf_tool))
    #        p = Tw[:3, 3]
    #        print(Tw)
    #    else:
    p = T[:3, 3]
    # ignore orientation
    x, y, z = p[0], p[1], p[2]
    a1, a2, a3 = links[0].dh.a, links[1].dh.a, links[2].dh.a
    tol = 1e-6
    reachable_pos = True
    reachable_neg = True

    # Theta 1
    # =======
    q1_pos = np.arctan2(y, x)
    q1_neg = np.arctan2(-y, -x)

    # Theta 3
    # =======
    # q3 two options, elbow up and elbow down
    # return solutions between in interval (-pi, pi)
    den = 2 * a2 * a3
    num = a1 ** 2 - a2 ** 2 - a3 ** 2 + x ** 2 + y ** 2 + z ** 2

    c1, s1 = np.cos(q1_pos), np.sin(q1_pos)
    c3 = (num - 2 * a1 * s1 * y - 2 * a1 * x * c1) / den

    if c3 > (1 + tol) or c3 < -(1 + tol):
        reachable_pos = False
    else:
        if c3 > (1 - tol) or c3 < -(1 - tol):
            # almost 1 or -1 => fix it
            c3 = np.sign(c3)

        s3 = np.sqrt(1 - c3 ** 2)
        q3_pos_a = np.arctan2(s3, c3)
        q3_pos_b = np.arctan2(-s3, c3)

    c1, s1 = np.cos(q1_neg), np.sin(q1_neg)
    c3 = (num - 2 * a1 * s1 * y - 2 * a1 * x * c1) / den

    if c3 > (1 + tol) or c3 < -(1 + tol):
        reachable_neg = False
    else:
        if c3 > (1 - tol) or c3 < -(1 - tol):
            # almost 1 or -1 => fix it
            c3 = np.sign(c3)

        s3 = np.sqrt(1 - c3 ** 2)
        q3_neg_a, q3_neg_b = np.arctan2(s3, c3), np.arctan2(-s3, c3)

    # Theta 2
    # =======
    L = np.sqrt(x ** 2 + y ** 2)
    if reachable_pos:
        c3, s3 = np.cos(q3_pos_b), np.sin(q3_pos_a)

        q2_pos_a = np.arctan2(
            (-a3 * s3 * (L - a1) + z * (a2 + a3 * c3)),
            (a3 * s3 * z + (L - a1) * (a2 + a3 * c3)),
        )
        q2_pos_b = np.arctan2(
            (a3 * s3 * (L - a1) + z * (a2 + a3 * c3)),
            (-a3 * s3 * z + (L - a1) * (a2 + a3 * c3)),
        )
    if reachable_neg:
        s3 = np.sin(q3_neg_a)
        c3 = np.cos(q3_neg_b)
        q2_neg_a = np.arctan2(
            (a3 * s3 * (L + a1) + z * (a2 + a3 * c3)),
            (a3 * s3 * z - (L + a1) * (a2 + a3 * c3)),
        )
        q2_neg_b = np.arctan2(
            (-a3 * s3 * (L + a1) + z * (a2 + a3 * c3)),
            (-a3 * s3 * z - (L + a1) * (a2 + a3 * c3)),
        )
    # q2_neg_a = -q2_neg_a
    # q2_neg_b = -q2_neg_b

    # 4 solutions
    # =========
    #        q_sol = [[q1_pos, q2_pos_a, q3_a],
    #                 [q1_pos, q2_pos_b, q3_b]]
    q_sol = []
    if reachable_pos:
        q_sol.append([q1_pos, q2_pos_a, q3_pos_a])
        q_sol.append([q1_pos, q2_pos_b, q3_pos_b])
    if reachable_neg:
        q_sol.append([q1_neg, q2_neg_a, q3_neg_a])
        q_sol.append([q1_neg, q2_neg_b, q3_neg_b])

    if reachable_pos or reachable_neg:
        return IKResult(True, q_sol)
    else:
        return IKResult(False)

