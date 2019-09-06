import numpy as np
import matplotlib.pyplot as plt

from acrobotics.util import get_default_axes3d, plot_reference_frame
from acrobotics.tool_examples import torch


def test_torch_model():

    fig, ax = get_default_axes3d([-0.10, 0.20], [0, 0.30], [-0.15, 0.15])
    plot_reference_frame(ax, torch.tf_tool_tip)
    torch.plot(ax, tf=np.eye(4), c="k")

    for tf in torch.tf_s:
        plot_reference_frame(ax, tf)
    plot_reference_frame(ax, torch.tf_tool_tip)

    # plt.show()
