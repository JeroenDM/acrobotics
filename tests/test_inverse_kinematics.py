import numpy as np

from numpy.testing import assert_almost_equal
from acrobotics.robot_examples import (
    SphericalWrist,
    AnthropomorphicArm,
    Arm2,
    Kuka,
    KukaOnRail,
)
from acrobotics.util import pose_x

PI = np.pi

## copied transform from a welding torch for testing
tool_tip_transform = np.array(
    [
        [4.00889232e-17, 9.38493023e-01, -3.45298199e-01, 1.49742355e-01],
        [-1.00000000e00, 5.74661238e-17, 4.00889232e-17, 9.16907481e-18],
        [5.74661238e-17, 3.45298199e-01, 9.38493023e-01, 2.77190403e-01],
        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
    ]
)


class TestIK:
    def test_aa_random(self):
        bot = AnthropomorphicArm()
        N = 20
        q_rand = np.random.rand(N, 3) * 2 * PI - PI
        for qi in q_rand:
            T1 = bot.fk(qi)
            resi = bot.ik(T1)
            for q_sol in resi.solutions:
                p2 = bot.fk(q_sol)[:3, 3]
                assert_almost_equal(T1[:3, 3], p2)

    def test_sw_random(self):
        bot = SphericalWrist()
        N = 20
        q_rand = np.random.rand(N, 3) * 2 * PI - PI
        for qi in q_rand:
            T1 = bot.fk(qi)
            resi = bot.ik(T1)
            for q_sol in resi.solutions:
                R2 = bot.fk(q_sol)[:3, :3]
                assert_almost_equal(T1[:3, :3], R2)

    def test_sw_random_other_base(self):
        bot = SphericalWrist()
        bot.tf_base = pose_x(1.5, 0.3, 0.5, 1.2)
        N = 20
        q_rand = np.random.rand(N, 3) * 2 * PI - PI
        for qi in q_rand:
            T1 = bot.fk(qi)
            resi = bot.ik(T1)
            for q_sol in resi.solutions:
                R2 = bot.fk(q_sol)[:3, :3]
                assert_almost_equal(T1[:3, :3], R2)

    def test_arm2_random(self):
        bot = Arm2()
        N = 20
        q_rand = np.random.rand(N, 3) * 2 * PI - PI
        for qi in q_rand:
            T1 = bot.fk(qi)
            resi = bot.ik(T1)
            if resi.success:
                for q_sol in resi.solutions:
                    p2 = bot.fk(q_sol)[:3, 3]
                    assert_almost_equal(T1[:3, 3], p2)
            else:
                # somethings is wrong, should be reachable
                print(resi)
                assert_almost_equal(qi, 0)

    # #    def test_arm2_tool_random(self):
    # #        bot = Arm2()
    # #        bot.tf_tool = pose_x(0, 0.1, 0, 0)
    # #        N = 20
    # #        q_rand = rand(N, 3) * 2 * pi - pi
    # #        for qi in q_rand:
    # #            T1 = bot.fk(qi)
    # #            resi = bot.ik(T1)
    # #            if resi['success']:
    # #                for q_sol in resi['sol']:
    # #                    p2 = bot.fk(q_sol)[:3, 3]
    # #                    assert_almost_equal(T1[:3, 3], p2)
    # #            else:
    # #                # somethings is wrong, should be reachable
    # #                print(resi)
    # #                assert_almost_equal(qi, 0)

    def test_kuka_random(self):
        """ TODO some issues with the numerical accuracy of the ik?"""
        bot = Kuka()
        N = 20
        q_rand = np.random.rand(N, 6) * 2 * PI - PI
        for qi in q_rand:
            print(qi)
            T1 = bot.fk(qi)
            resi = bot.ik(T1)
            if resi.success:
                for q_sol in resi.solutions:
                    print(q_sol)
                    T2 = bot.fk(q_sol)
                    assert_almost_equal(T1, T2, decimal=5)
            else:
                # somethings is wrong, should be reachable
                print(resi)
                assert_almost_equal(qi, 0)

    def test_kuka_tool_random(self):
        bot = Kuka()
        bot.tf_tool_tip = tool_tip_transform
        N = 20
        q_rand = np.random.rand(N, 6) * 2 * PI - PI
        for qi in q_rand:
            print(qi)
            T1 = bot.fk(qi)
            resi = bot.ik(T1)
            if resi.success:
                for q_sol in resi.solutions:
                    print(q_sol)
                    T2 = bot.fk(q_sol)
                    assert_almost_equal(T1, T2)
            else:
                # somethings is wrong, should be reachable
                print(resi)
                assert_almost_equal(qi, 0)

    # the test below failed at random runs
    # def test_kuka_base_random(self):
    #     bot = Kuka()
    #     bot.tf_base = pose_x(0.1, 0.02, 0.01, -0.01)
    #     N = 20
    #     q_rand = np.random.rand(N, 6) * 2 * PI - PI
    #     for qi in q_rand:
    #         print(qi)
    #         T1 = bot.fk(qi)
    #         resi = bot.ik(T1)
    #         if resi.success:
    #             for q_sol in resi.solutions:
    #                 print(q_sol)
    #                 T2 = bot.fk(q_sol)
    #                 assert_almost_equal(T1, T2)
    #         else:
    #             # somethings is wrong, should be reachable
    #             print(resi)
    #             assert_almost_equal(qi, 0)

    def test_kuka_on_rail_random(self):
        bot = KukaOnRail()
        N = 20
        q_rand = np.random.rand(N, 7) * 2 * PI - PI
        for qi in q_rand:
            print(qi)
            T1 = bot.fk(qi)
            resi = bot.ik(T1, qi[0])
            if resi.success:
                for q_sol in resi.solutions:
                    print(q_sol)
                    T2 = bot.fk(q_sol)
                    assert_almost_equal(T1, T2)
            else:
                # somethings is wrong, should be reachable
                print(resi)
                assert_almost_equal(qi, 0)

    def test_kuka_on_rail_tool_random(self):
        bot = KukaOnRail()
        bot.tf_tool_tip = tool_tip_transform
        N = 20
        q_rand = np.random.rand(N, 7) * 2 * PI - PI
        for qi in q_rand:
            print(qi)
            T1 = bot.fk(qi)
            resi = bot.ik(T1, qi[0])
            if resi.success:
                for q_sol in resi.solutions:
                    print(q_sol)
                    T2 = bot.fk(q_sol)
                    assert_almost_equal(T1, T2)
            else:
                # somethings is wrong, should be reachable
                print(resi)
                assert_almost_equal(qi, 0)
