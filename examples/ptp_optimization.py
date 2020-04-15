import numpy as np

# ======================================================
# Create planning scene
# ======================================================
from acrobotics.robot_examples import Kuka
from acrobotics.geometry import Scene
from acrobotics.shapes import Box, Cylinder
from acrobotics.robot import Tool
from acrolib.geometry import translation, pose_y
from acrobotics.urdfio import import_urdf

# create the robot
robot = Kuka()
vacuum_tool = Tool(
    [Cylinder(0.025, 0.15)], [translation(0, 0, 0.075)], translation(0, 0, 0.15)
)
robot.tool = vacuum_tool
robot.tf_base = translation(0, 0, 0.1)

# load the bin model from a file and move it in the right place
scene = import_urdf("examples/bin.urdf")
scene.translate(0.1, -0.7, -0.025)

# add a nice table to the whole setup
table = Box(1.2, 1.2, 0.05)
T_table = translation(0.4, -0.3, -0.04)
scene.add([table], [T_table])

# create a picking pose in the bin
# the z-axis should point down, as the tool's z axis
# sticks out of the tool center point
T_pick = pose_y(np.pi, 0.1, -0.7, 0.1)

# check if we can reach this picking pose
ik_sol = robot.ik(T_pick)
if ik_sol.success:
    print(
        "Picking pose can be reached by {} configurations".format(len(ik_sol.solutions))
    )
else:
    print("Failed to reach the picking pose.")


q_start = np.array([0.5, 1.5, -0.3, 0, 0, 0])
# q_goal = np.array([-2.5, 1.5, 0.3, 0, 0, 0])
q_goals = ik_sol.solutions

# T_goal = robot.fk(q_goal)

# ======================================================
# Setup optimization problem
# ======================================================
N = 30  # number of points along the path
from acrobotics.planning.optimization_based import create_cc
import casadi

opti = casadi.Opti()

q = opti.variable(N, 6)  #  joint variables along path

# objective function
V = 0
for i in range(1, N):
    for k in range(robot.ndof):
        V += (q[i, k] - q[i - 1, k]) ** 2

opti.minimize(V)

opti.subject_to(q[0, :] == q_start.reshape(1, 6))

# opti.subject_to(q[-1, :] == q_goal.reshape(1, 6))
opti.subject_to(robot.fk_casadi(q[-1, :]) == T_pick)

opti.subject_to(create_cc(opti, robot, scene, q))

opti.solver("ipopt")

q_path_init = np.linspace(q_start, q_goals[2], N)
# opti.set_initial(q, q_path_init)

# ======================================================
# Solve optimization problem
# ======================================================
try:
    solution = opti.solve()
    q_sol = solution.value(q)
except RuntimeError as e:
    print(e)
    q_sol = opti.debug.value(q)

print([robot.is_in_collision(q, scene) for q in q_sol])
# q_sol = np.linspace(q_start, q_goals[2], N)
# ======================================================
# Animate path and planning scene
# ======================================================
import matplotlib.pyplot as plt
from acrolib.plotting import get_default_axes3d, plot_reference_frame

fig, ax = get_default_axes3d([-0.8, 0.8], [-0.8, 0.8], [-0.2, 1.4])
ax.set_axis_off()
ax.view_init(elev=31, azim=-15)

plot_reference_frame(ax, tf=T_pick, arrow_length=0.4)

scene.plot(ax, c="green")
robot.animate_path(fig, ax, q_sol)
robot.plot_kinematics(ax, q_start, c="k")

# robot.animation.save("examples/robot_animation.gif", writer="imagemagick", fps=10)
plt.show()
