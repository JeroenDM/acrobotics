import numpy as np
import matplotlib.pyplot as plt

from acrolib.quaternion import Quaternion
from acrobotics.path.tolerance import NoTolerance, SymmetricTolerance, Tolerance
from acrobotics.path.path_pt import TolEulerPt
from acrobotics.path.factory import create_line
from acrobotics.util import get_default_axes3d, plot_reference_frame, translation
from acrobotics.geometry import Scene

from acrobotics.robot_examples import Kuka
from acrobotics.tool_examples import torch2

from acrobotics.planning.types import SolveMethod, PlanningSetup, CostFuntionType
from acrobotics.path.sampling import SampleMethod, SearchStrategy
from acrobotics.planning.sampling_based import sampling_based_solve
from acrobotics.planning.settings import SamplingSetting, SolverSettings


from acrobotics.irlio import import_irl_paths
from acrobotics.urdfio import import_urdf

robot = Kuka()
robot.tool = torch2
robot.tf_base = translation(0, 0, 0.4)

paths = import_irl_paths("kingpin.irl")
scene = import_urdf("kingpin.urdf")


settings = SamplingSetting(
    SearchStrategy.INCREMENTAL,
    sample_method=SampleMethod.random_uniform,
    num_samples=300,
    iterations=1,
    tolerance_reduction_factor=2,
)

solve_set = SolverSettings(
    SolveMethod.sampling_based, CostFuntionType.l1_norm, sampling_settings=settings,
)

solutions = []
for path in paths:
    setup = PlanningSetup(robot, path, scene)
    sol = sampling_based_solve(setup, solve_set)
    solutions.append(sol)


sol_path = []
for s in solutions:
    sol_path += s.joint_positions

# N = 10
# qs = np.linspace(-np.pi, np.pi, N)
# sweep_path = np.zeros((6 * N, 6))
# for i in range(6):
#     sweep_path[(N * i) : (N * (i + 1)), i] = qs


fig, ax = get_default_axes3d()
# robot.plot(ax, [0, 0, 0, 0, 0, 0], c="k")
plot_reference_frame(ax, tf=robot.fk([0, 1.5, 0, 0, 0, 0]))
# scene.plot(ax, c="g")
for path in paths:
    for pt in path:
        plot_reference_frame(ax, pt.transformation_matrix)
robot.animate_path(fig, ax, sol_path)
# robot.animate_path(fig, ax, sweep_path)
plt.show()

np.savetxt("data/solution11.txt", np.array(sol_path))

# # ========================================================
# # optimization based path smoothing
# # ========================================================
# from acrobotics.planning.solver import solve
# from acrobotics.planning.settings import OptSettings

# s2 = SolverSettings(
#     SolveMethod.optimization_based,
#     CostFuntionType.sum_squared,
#     opt_settings=OptSettings(q_init=np.array(sol.joint_positions)),
# )

# sol2 = solve(setup, s2)


# fig, ax = get_default_axes3d()
# # plot_reference_frame(ax, pt_1.transformation_matrix)

# robot.plot(ax, [0, 1.5, 0, 0, 0, 0], c="k")
# plot_reference_frame(ax, tf=robot.fk([0, 1.5, 0, 0, 0, 0]))
# scene.plot(ax, c="g")

# for pt in path:
#     plot_reference_frame(ax, pt.transformation_matrix)

# print(sol)
# robot.animate_path(fig, ax, sol2.joint_positions)

# plt.show()
