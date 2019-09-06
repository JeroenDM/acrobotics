import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


def rot_x(alfa):
    return np.array(
        [[1, 0, 0], [0, np.cos(alfa), -np.sin(alfa)], [0, np.sin(alfa), np.cos(alfa)]]
    )


def rot_y(alfa):
    return np.array(
        [[np.cos(alfa), 0, np.sin(alfa)], [0, 1, 0], [-np.sin(alfa), 0, np.cos(alfa)]]
    )


def rot_z(alfa):
    return np.array(
        [[np.cos(alfa), -np.sin(alfa), 0], [np.sin(alfa), np.cos(alfa), 0], [0, 0, 1]]
    )


def translation(x, y, z):
    return np.array(
        [[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]], dtype="float64"
    )


def pose_x(alfa, x, y, z):
    """ Homogenous transform with rotation around x-axis and translation. """
    return np.array(
        [
            [1, 0, 0, x],
            [0, np.cos(alfa), -np.sin(alfa), y],
            [0, np.sin(alfa), np.cos(alfa), z],
            [0, 0, 0, 1],
        ]
    )


def pose_y(alfa, x, y, z):
    """ Homogenous transform with rotation around y-axis and translation. """
    return np.array(
        [
            [np.cos(alfa), 0, np.sin(alfa), x],
            [0, 1, 0, y],
            [-np.sin(alfa), 0, np.cos(alfa), z],
            [0, 0, 0, 1],
        ]
    )


def pose_z(alfa, x, y, z):
    """ Homogenous transform with rotation around z-axis and translation. """
    return np.array(
        [
            [np.cos(alfa), -np.sin(alfa), 0, x],
            [np.sin(alfa), np.cos(alfa), 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1],
        ]
    )


def get_default_axes3d(xlim=[-1, 1], ylim=[-1, 1], zlim=[-1, 1]):
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.set_xlim3d(xlim)
    ax.set_ylim3d(ylim)
    ax.set_zlim3d(zlim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    return fig, ax


def plot_reference_frame(ax, tf=None, arrow_length=0.2):
    """ Plot xyz-axes on axes3d object

    Parameters
    ----------
    ax : mpl_toolkits.mplot3d.Axes3D
        Axes object for 3D plotting.
    tf : np.array of float
        Transform to specify location of axes. Plots in origin if None.
    l : float
        The length of the axes plotted.
    """
    l = arrow_length
    x_axis = np.array([[0, l], [0, 0], [0, 0]])
    y_axis = np.array([[0, 0], [0, l], [0, 0]])
    z_axis = np.array([[0, 0], [0, 0], [0, l]])

    if tf is not None:
        # rotation
        x_axis = np.dot(tf[:3, :3], x_axis)
        y_axis = np.dot(tf[:3, :3], y_axis)
        z_axis = np.dot(tf[:3, :3], z_axis)
        # translation [:, None] numpian way to change shape (add axis)
        x_axis = x_axis + tf[:3, 3][:, None]
        y_axis = y_axis + tf[:3, 3][:, None]
        z_axis = z_axis + tf[:3, 3][:, None]

    ax.plot(x_axis[0], x_axis[1], x_axis[2], "-", c="r")
    ax.plot(y_axis[0], y_axis[1], y_axis[2], "-", c="g")
    ax.plot(z_axis[0], z_axis[1], z_axis[2], "-", c="b")
