import numpy as np
from numpy.testing import assert_almost_equal

from acrolib.quaternion import Quaternion

from acrobotics.path.factory import create_line, create_arc
from acrobotics.path.path_pt import TolPositionPt, TolEulerPt
from acrobotics.path.tolerance import Tolerance, SymmetricTolerance, NoTolerance


class TestPathFactory:
    def test_create_line(self):
        pos = [1, 2, 3]
        tol = [
            SymmetricTolerance(1, 10),
            SymmetricTolerance(1, 10),
            SymmetricTolerance(1, 10),
        ]
        start_pt = TolPositionPt(pos, Quaternion.random(), tol)
        path = create_line(start_pt, np.array([2, 2, 3]), 3)
        assert path[-1].pos[0] == 2
        assert_almost_equal(path[1].pos[0], 1.5)
        assert path[-1].pos[1] == pos[1]
        assert path[-1].pos[1] == pos[1]
        assert path[-1].pos[2] == pos[2]
        assert path[-1].pos[2] == pos[2]
        assert path[-1].pos[0] == pos[0] + 1
        assert path[-1].pos[0] == pos[0] + 1

    def test_create_arc(self):
        pos = [1, 0, 3]
        pos_tol = [NoTolerance(), NoTolerance(), NoTolerance()]
        rot_tol = [
            SymmetricTolerance(0.5, 10),
            SymmetricTolerance(0.5, 10),
            SymmetricTolerance(0.5, 10),
        ]
        start_pt = TolEulerPt(pos, Quaternion(), pos_tol, rot_tol)
        path = create_arc(start_pt, np.zeros(3), np.array([0, 0, 1]), np.pi / 2, 3)
        assert len(path) == 3
        p1, p2, p3 = path[0], path[1], path[2]

        assert_almost_equal(p1.pos[0], 1)
        assert_almost_equal(p1.pos[1], 0)
        assert_almost_equal(p1.pos[2], 3)

        assert_almost_equal(p2.pos[0], np.sqrt(2) / 2)
        assert_almost_equal(p2.pos[1], np.sqrt(2) / 2)
        assert_almost_equal(p2.pos[2], 3)

        assert_almost_equal(p3.pos[0], 0)
        assert_almost_equal(p3.pos[1], 1)
        assert_almost_equal(p3.pos[2], 3)
