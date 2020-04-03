import numpy as np
import matplotlib.pyplot as plt

from acrolib.plotting import get_default_axes3d, plot_reference_frame
from acrobotics.tool_examples import torch, torch2, torch3


def test_torch_model():

    fig, ax = get_default_axes3d([-0.10, 0.20], [0, 0.30], [-0.15, 0.15])
    plot_reference_frame(ax, torch.tf_tool_tip)
    torch.plot(ax, tf=np.eye(4), c="k")

    for tf in torch.tf_s:
        plot_reference_frame(ax, tf)
    plot_reference_frame(ax, torch.tf_tool_tip)

    # plt.show()


def test_torch_model_2():

    fig, ax = get_default_axes3d([-0.10, 0.20], [0, 0.30], [-0.15, 0.15])
    plot_reference_frame(ax, torch2.tf_tool_tip)
    torch2.plot(ax, tf=np.eye(4), c="k")

    for tf in torch2.tf_s:
        plot_reference_frame(ax, tf)
    plot_reference_frame(ax, torch2.tf_tool_tip)

    # plt.show()


def test_torch_model_3():

    fig, ax = get_default_axes3d([-0.10, 0.20], [0, 0.30], [-0.15, 0.15])
    plot_reference_frame(ax, torch3.tf_tool_tip)
    torch3.plot(ax, tf=np.eye(4), c="k")

    for tf in torch3.tf_s:
        plot_reference_frame(ax, tf)
    plot_reference_frame(ax, torch3.tf_tool_tip)

    # plt.show()
