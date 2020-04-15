"""
Simple motion to a configuration space goal, around an obstacle.

Formulate it as a non-linear optimization problem using casadi.

Add collision constraints as explained in the paper:
"Time-Optimal Path Following for Robots
with Object Collision Avoidance using Lagrangian Duality"
By Debrouwere et al.

Also, I'm sorry acrobotics requires so much import statements.
"""
import numpy as np

# ======================================================
# Create planning scene
# ======================================================
from acrobotics.robot_examples import Kuka
from acrobotics.geometry import Scene
from acrobotics.shapes import Box
from acrolib.geometry import translation

robot = Kuka()

table = Box(2, 2, 0.1)
T_table = translation(0, 0, -0.2)

obstacle = Box(0.2, 0.2, 0.8)
T_obs = translation(0, 0.5, 0.3)

scene = Scene([table, obstacle], [T_table, T_obs])

q_start = np.array([0.5, 1.5, -0.3, 0, 0, 0])
q_goal = np.array([2.5, 1.5, -0.3, 0, 0, 0])

# ======================================================
# Setup optimization problem
# ======================================================
N = 10  # number of points along the path
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
opti.subject_to(q[-1, :] == q_goal.reshape(1, 6))

opti.subject_to(create_cc(opti, robot, scene, q))

opti.solver("ipopt")

# without a proper initial guess, the robot jumps through
# the obstacle with the current objective function
q_path_init = np.linspace(q_start, q_goal, N)
opti.set_initial(q, q_path_init)

# ======================================================
# Solve optimization problem
# ======================================================
try:
    solution = opti.solve()
    q_sol = solution.value(q)
except RuntimeError as e:
    print(e)
    q_sol = opti.debug.value(q)

# check if to robot is indeed not in collision along the path
# it turns out this is not always the case, so there is an unkown
# issue with the collision constraints
print([robot.is_in_collision(q, scene) for q in q_sol])

# we can also visualize the initial guess
# q_sol = np.linspace(q_start, q_goals[2], N)

# ======================================================
# Animate path and planning scene
# ======================================================
import matplotlib.pyplot as plt
from acrolib.plotting import get_default_axes3d, plot_reference_frame

fig, ax = get_default_axes3d([-0.8, 0.8], [-0.8, 0.8], [-0.2, 1.4])
ax.set_axis_off()
ax.view_init(elev=31, azim=-15)

scene.plot(ax, c="green")
robot.animate_path(fig, ax, q_sol)
# robot.plot_kinematics(ax, q_start, c="k")

# robot.animation.save("examples/robot_animation.gif", writer="imagemagick", fps=10)
plt.show()
