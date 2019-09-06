from numpy import cos, sin, array, eye


class FKImplementations:
    # compare with analytic solution from the book
    # "Robotics: modelling, planning and control"
    @staticmethod
    def fk_PlanarArm(q, links):
        a1, a2, a3 = links[0].dh.a, links[1].dh.a, links[2].dh.a
        c123 = cos(q[0] + q[1] + q[2])
        s123 = sin(q[0] + q[1] + q[2])
        T = eye(4)
        T[0, 0] = c123
        T[0, 1] = -s123
        T[1, 0] = s123
        T[1, 1] = c123
        T[0, 3] = a1 * cos(q[0]) + a2 * cos(q[0] + q[1]) + a3 * c123
        T[1, 3] = a1 * sin(q[0]) + a2 * sin(q[0] + q[1]) + a3 * s123
        return T

    @staticmethod
    def fk_SphericalArm(q, links):
        d2, d3 = links[1].dh.d, q[2]
        c1 = cos(q[0])
        s1 = sin(q[0])
        c2 = cos(q[1])
        s2 = sin(q[1])
        T = eye(4)
        T[0, 0:3] = array([c1 * c2, -s1, c1 * s2])
        T[0, 3] = c1 * s2 * d3 - s1 * d2
        T[1, 0:3] = array([s1 * c2, c1, s1 * s2])
        T[1, 3] = s1 * s2 * d3 + c1 * d2
        T[2, 0:3] = array([-s2, 0, c2])
        T[2, 3] = c2 * d3
        return T

    @staticmethod
    def fk_AnthropomorphicArm(q, links):
        a2, a3 = links[1].dh.a, links[2].dh.a
        c1 = cos(q[0])
        s1 = sin(q[0])
        c2 = cos(q[1])
        s2 = sin(q[1])
        c23 = cos(q[1] + q[2])
        s23 = sin(q[1] + q[2])
        T = eye(4)
        T[0, 0:3] = array([c1 * c23, -c1 * s23, s1])
        T[0, 3] = c1 * (a2 * c2 + a3 * c23)
        T[1, 0:3] = array([s1 * c23, -s1 * s23, -c1])
        T[1, 3] = s1 * (a2 * c2 + a3 * c23)
        T[2, 0:3] = array([s23, c23, 0])
        T[2, 3] = a2 * s2 + a3 * s23
        return T

    @staticmethod
    def fk_Arm2(q, links):
        a1, a2, a3 = links[0].dh.a, links[1].dh.a, links[2].dh.a
        c1 = cos(q[0])
        s1 = sin(q[0])
        c2 = cos(q[1])
        s2 = sin(q[1])
        c23 = cos(q[1] + q[2])
        s23 = sin(q[1] + q[2])
        T = eye(4)
        T[0, 0:3] = array([c1 * c23, s1, c1 * s23])
        T[1, 0:3] = array([s1 * c23, -c1, s1 * s23])
        T[2, 0:3] = array([s23, 0, -c23])
        T[0, 3] = c1 * (a1 + a2 * c2 + a3 * c23)
        T[1, 3] = s1 * (a1 + a2 * c2 + a3 * c23)
        T[2, 3] = a2 * s2 + a3 * s23
        return T

    @staticmethod
    def fk_kuka(q):
        q1, q2, q3, q4, q5, q6 = q
        a1 = 0.18
        a2 = 0.6
        d4 = 0.62
        d6 = 0.115
        T = array(
            [
                [
                    (
                        (sin(q1) * sin(q4) + cos(q1) * cos(q4) * cos(q2 + q3)) * cos(q5)
                        - sin(q5) * sin(q2 + q3) * cos(q1)
                    )
                    * cos(q6)
                    + (sin(q1) * cos(q4) - sin(q4) * cos(q1) * cos(q2 + q3)) * sin(q6),
                    -(
                        (sin(q1) * sin(q4) + cos(q1) * cos(q4) * cos(q2 + q3)) * cos(q5)
                        - sin(q5) * sin(q2 + q3) * cos(q1)
                    )
                    * sin(q6)
                    + (sin(q1) * cos(q4) - sin(q4) * cos(q1) * cos(q2 + q3)) * cos(q6),
                    (sin(q1) * sin(q4) + cos(q1) * cos(q4) * cos(q2 + q3)) * sin(q5)
                    + sin(q2 + q3) * cos(q1) * cos(q5),
                    a1 * cos(q1)
                    + a2 * cos(q1) * cos(q2)
                    + d4 * sin(q2 + q3) * cos(q1)
                    + d6
                    * (
                        (sin(q1) * sin(q4) + cos(q1) * cos(q4) * cos(q2 + q3)) * sin(q5)
                        + sin(q2 + q3) * cos(q1) * cos(q5)
                    ),
                ],
                [
                    (
                        (sin(q1) * cos(q4) * cos(q2 + q3) - sin(q4) * cos(q1)) * cos(q5)
                        - sin(q1) * sin(q5) * sin(q2 + q3)
                    )
                    * cos(q6)
                    - (sin(q1) * sin(q4) * cos(q2 + q3) + cos(q1) * cos(q4)) * sin(q6),
                    -(
                        (sin(q1) * cos(q4) * cos(q2 + q3) - sin(q4) * cos(q1)) * cos(q5)
                        - sin(q1) * sin(q5) * sin(q2 + q3)
                    )
                    * sin(q6)
                    - (sin(q1) * sin(q4) * cos(q2 + q3) + cos(q1) * cos(q4)) * cos(q6),
                    (sin(q1) * cos(q4) * cos(q2 + q3) - sin(q4) * cos(q1)) * sin(q5)
                    + sin(q1) * sin(q2 + q3) * cos(q5),
                    a1 * sin(q1)
                    + a2 * sin(q1) * cos(q2)
                    + d4 * sin(q1) * sin(q2 + q3)
                    + d6
                    * (
                        (sin(q1) * cos(q4) * cos(q2 + q3) - sin(q4) * cos(q1)) * sin(q5)
                        + sin(q1) * sin(q2 + q3) * cos(q5)
                    ),
                ],
                [
                    (sin(q5) * cos(q2 + q3) + sin(q2 + q3) * cos(q4) * cos(q5))
                    * cos(q6)
                    - sin(q4) * sin(q6) * sin(q2 + q3),
                    -(sin(q5) * cos(q2 + q3) + sin(q2 + q3) * cos(q4) * cos(q5))
                    * sin(q6)
                    - sin(q4) * sin(q2 + q3) * cos(q6),
                    sin(q5) * sin(q2 + q3) * cos(q4) - cos(q5) * cos(q2 + q3),
                    a2 * sin(q2)
                    - d4 * cos(q2 + q3)
                    + d6 * (sin(q5) * sin(q2 + q3) * cos(q4) - cos(q5) * cos(q2 + q3)),
                ],
                [0, 0, 0, 1],
            ]
        )
        return T
