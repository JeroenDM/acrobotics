"""
Getting started example from readme.txt

This example shows how to create a planning scene,
visualize a robot and check for collision between the two.
Also, forward and inverse kinematics are nice.
"""
import acrobotics as ab

# ======================================================
# Get a ready to go robot implementation
# ======================================================
robot = ab.Kuka()

# ======================================================
# Create planning scene
# ======================================================
# A transform matrix is represented by a 4x4 numpy array
# Here we create one using a helper function for readability
from acrolib.geometry import translation

table = ab.Box(2, 2, 0.1)
T_table = translation(0, 0, -0.2)

obstacle = ab.Box(0.2, 0.2, 1.5)
T_obs = translation(0, 0.5, 0.55)

scene = ab.Scene([table, obstacle], [T_table, T_obs])

# ======================================================
# Linear interpolation path from start to goal
# ======================================================
import numpy as np

q_start = np.array([0.5, 1.5, -0.3, 0, 0, 0])
q_goal = np.array([2.5, 1.5, 0.3, 0, 0, 0])
q_path = np.linspace(q_start, q_goal, 10)

# ======================================================
# Show which parts of the path are in collision
# ======================================================
print([robot.is_in_collision(q, scene) for q in q_path])

# ======================================================
# Calculate forward and inverse kinematics
# ======================================================
# forward kinematics are available by default
T_fk = robot.fk([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

# inverse kinematics are implemented for specific robots
ik_solution = robot.ik(T_fk)

print(f"Inverse kinematics successful? {ik_solution.success}")
for q in ik_solution.solutions:
    print(q)

# ======================================================
# Animate path and planning scene
# ======================================================
import matplotlib.pyplot as plt
from acrolib.plotting import get_default_axes3d

fig, ax = get_default_axes3d([-0.8, 0.8], [-0.8, 0.8], [-0.2, 1.4])
ax.set_axis_off()
ax.view_init(elev=31, azim=-15)

scene.plot(ax, c="green")
robot.animate_path(fig, ax, q_path)

# robot.animation.save("examples/robot_animation.gif", writer="imagemagick", fps=10)
plt.show()

