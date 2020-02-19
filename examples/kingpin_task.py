import numpy as np
import matplotlib.pyplot as plt

from acrolib.quaternion import Quaternion
from acrobotics.path.tolerance import NoTolerance, SymmetricTolerance, Tolerance
from acrobotics.path.path_pt import TolEulerPt
from acrobotics.path.factory import create_line
from acrobotics.util import get_default_axes3d, plot_reference_frame
from acrobotics.geometry import Scene

from acrobotics.robot_examples import Kuka
from acrobotics.tool_examples import torch2

from acrobotics.planning.types import SolveMethod, PlanningSetup, CostFuntionType
from acrobotics.path.sampling import SampleMethod, SearchStrategy
from acrobotics.planning.sampling_based import sampling_based_solve
from acrobotics.planning.settings import SamplingSetting, SolverSettings


from acrobotics.irlio import parse_file, tf_interpolations
from acrobotics.urdfio import import_urdf

robot = Kuka()
robot.tool = torch2

task = parse_file("kingpin.irl")
scene = import_urdf("kingpin.urdf")


tf_1 = task["variables"]["P2"]
tf_2 = task["variables"]["P4"]
cons1 = task["constraints"]["c-weld"]
print(cons1)
# q1 = [tf_1["xyzw"][3]] + tf_1["xyzw"][:3]

# pos_tol = [NoTolerance()] * 3
# rot_tol = [NoTolerance(), NoTolerance(), SymmetricTolerance(1.5, 10)]
# pt_1 = TolEulerPt(tf_1[:3, 3], Quaternion(matrix=tf_1), pos_tol, rot_tol)

# path = create_line(pt_1, np.array(tf_2[:3, 3]), 10)


def parse_constraints(c):
    tol = []
    for lower, upper in zip(c["min"], c["max"]):
        if lower == upper:
            tol.append(NoTolerance())
        else:
            tol.append(Tolerance(lower, upper, 20))
    return tol


trans, rots = tf_interpolations(tf_1, tf_2, 10)
path = []
for p, R in zip(trans, rots):
    pos_tol = [NoTolerance()] * 3
    rot_tol = parse_constraints(cons1)
    path.append(TolEulerPt(p, Quaternion(matrix=R.as_matrix()), pos_tol, rot_tol))


setup = PlanningSetup(robot, path, scene)
settings = SamplingSetting(
    SearchStrategy.INCREMENTAL,
    sample_method=SampleMethod.random_uniform,
    num_samples=100,
    iterations=1,
    tolerance_reduction_factor=2,
)

solve_set = SolverSettings(
    SolveMethod.sampling_based, CostFuntionType.l1_norm, sampling_settings=settings,
)

sol = sampling_based_solve(setup, solve_set)

fig, ax = get_default_axes3d()
# plot_reference_frame(ax, pt_1.transformation_matrix)

robot.plot(ax, [0, 1.5, 0, 0, 0, 0], c="k")
plot_reference_frame(ax, tf=robot.fk([0, 1.5, 0, 0, 0, 0]))
scene.plot(ax, c="g")

for pt in path:
    plot_reference_frame(ax, pt.transformation_matrix)

print(sol)
robot.animate_path(fig, ax, sol.joint_positions)

plt.show()
