"""
Define a simplified model of a welding torch.
"""
import numpy as np
from acrolib.geometry import rot_x, rot_y, rot_z

from .shapes import Box, Cylinder
from .geometry import Scene
from .robot import Tool

# ==============================================================================
# Dimensions
# ==============================================================================
angle1 = 11.3 * np.pi / 180  # torch relative to base rotation
angle2 = -31.5 * np.pi / 180  # torch tip angle

pos_data = np.array(
    [
        [0.035, 0, 0],
        [-0.02, 0.15, 0],
        [0.085, 0.075, 0],
        [0.15, 0.15, 0],
        [0.25, 0.15 + np.sin(angle2) * 0.05, 0],
    ]
)

s = [
    Box(0.08, 0.08, 0.08),
    Box(0.18, 0.075, 0.075),
    Box(0.03, 0.22, 0.07),
    Box(0.10, 0.025, 0.025),
    Box(0.10, 0.025, 0.025),
]

z_offset = 0.02  # tool stick-put + offset

# ==============================================================================
# Create shape transforms relative to tool base
# ==============================================================================
tf1, tf2, tf3, tf4, tf5 = np.eye(4), np.eye(4), np.eye(4), np.eye(4), np.eye(4)
tool_tip = np.eye(4)

# translaten relative to first shape
tf1[:3, 3] = pos_data[0]
tf2[:3, 3] = pos_data[1]
tf3[:3, 3] = pos_data[2]
tf4[:3, 3] = pos_data[3]
tf5[:3, 3] = pos_data[4]
R2 = rot_z(angle2)
tf5[:3, :3] = R2
tool_tip[:3, 3] = pos_data[4] + np.dot(R2, np.array([0.05 + z_offset, 0, 0]))
tool_tip[:3, :3] = R2

# rotate links 2-5 relative to base 1
tf_rotate = np.eye(4)
tf_rotate[:3, :3] = rot_z(angle1)
tf2 = np.dot(tf_rotate, tf2)
tf3 = np.dot(tf_rotate, tf3)
tf4 = np.dot(tf_rotate, tf4)
tf5 = np.dot(tf_rotate, tf5)
tool_tip = np.dot(tf_rotate, tool_tip)

# point z-axis out of tool tip
tool_tip[:3, :3] = np.dot(tool_tip[:3, :3], rot_y(np.pi / 2))


tfs = [tf1, tf2, tf3, tf4, tf5]

# ==============================================================================
# Tool 1
# ==============================================================================
tf_tool = np.eye(4)
tf_tool[:3, :3] = np.dot(rot_y(-np.pi / 2), rot_x(-np.pi / 2))

tool_tip = np.dot(tf_tool, tool_tip)
for i in range(len(tfs)):
    tfs[i] = np.dot(tf_tool, tfs[i])

torch = Tool(s, tfs, tool_tip)

# ==============================================================================
# Tool 2 with a "rounded" tip
# ==============================================================================
tf6 = np.copy(tf5)
tf6[:3, 3] = tf6[:3, 3] + tf5[:3, :3] @ np.array([0.045, 0, 0])


s6 = Box(0.01, 0.018, 0.018)
tfs_2 = [tf1, tf2, tf3, tf4, tf5, tf6]
tfs_2 = [tf_tool @ tf for tf in tfs_2]

shapes_2 = s.copy()
shapes_2.append(s6)
shapes_2[-2] = Box(0.09, 0.025, 0.025)
tfs_2[-2][:3, 3] = tfs_2[-2][:3, 3] + tfs_2[-2][:3, :3] @ np.array([-0.005, 0, 0])

torch2 = Tool(shapes_2, tfs_2, tool_tip)

# ==============================================================================
# Tool 3: replace last two tip shapes with cylinders
# ==============================================================================


shapes_3 = [
    Box(0.08, 0.08, 0.08),
    Box(0.18, 0.075, 0.075),
    Box(0.03, 0.22, 0.07),
    Box(0.10, 0.025, 0.025),
    Cylinder(0.025 / 2, 0.09),  # Box(0.10, 0.025, 0.025),
    Cylinder(0.018 / 2, 0.01),  # Box(0.01, 0.018, 0.018),
]

tfs_3 = [np.copy(tf) for tf in tfs_2]


# cylinders along z-axis, boxes where along x-axis
R_adjust = rot_y(np.pi / 2)
tfs_3[-2][:3, :3] = tfs_3[-2][:3, :3] @ R_adjust
tfs_3[-1][:3, :3] = tfs_3[-1][:3, :3] @ R_adjust


torch3 = Tool(shapes_3, tfs_3, tool_tip)
