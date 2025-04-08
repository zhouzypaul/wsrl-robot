import jax
import matplotlib.pyplot as plt
import numpy as np
from absl import flags
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from serl_launcher.common.env_common import calc_return_to_go

FLAGS = flags.FLAGS


def make_single_trajectory_mc_visual(
    q_estimates,
    mc_returns,
):
    fig, axs = plt.subplots(2, 1, figsize=(8, 15))
    canvas = FigureCanvas(fig)
    plt.xlim([0, len(q_estimates[0])])

    # double Q
    if len(q_estimates.shape) == 2:
        axs[0].plot(q_estimates[0, :], linestyle="--", marker="o")
        axs[0].plot(q_estimates[1, :], linestyle="--", marker="o")
    else:
        axs[0].plot(q_estimates, linestyle="--", marker="o")
    axs[0].set_ylabel("q values")

    if len(mc_returns.shape) == 2:
        axs[1].plot(mc_returns[0, :], linestyle="--", marker="o")
        axs[1].plot(mc_returns[1, :], linestyle="--", marker="o")
    else:
        axs[1].plot(mc_returns, linestyle="--", marker="o")
    axs[1].set_ylabel("mc_returns")

    plt.tight_layout()

    canvas.draw()
    out_image = np.frombuffer(canvas.buffer_rgba(), dtype="uint8")
    out_image = out_image.reshape(fig.canvas.get_width_height()[::-1] + (4,))

    return out_image


def mc_q_visualization(trajs, agent, discount=0.99, seed=0, exp_name="peg_insertion"):
    def _batch_dicts(stat):
        """stat is a list of dict, turn it into a dict of list"""
        d = {}
        for k in stat[0].keys():
            d[k] = np.array([s[k] for s in stat])
        return d

    rng = jax.random.PRNGKey(seed)
    n_trajs = len(trajs)
    visualization_images = []

    # for each trajectory
    for i in range(n_trajs):
        observations = np.array(trajs[i]["observations"])
        actions = np.array(trajs[i]["actions"])
        rewards = np.array(trajs[i]["rewards"])
        if "masks" in trajs[i]:
            masks = np.array(trajs[i]["masks"])
        else:
            masks = np.array([not d for d in trajs[i]["dones"]])

        q_pred = agent.forward_critic(observations, actions, rng=None, train=False)

        mc_returns = calc_return_to_go(
            exp_name,
            rewards,
            masks,
            discount,
        )
        visualization_images.append(
            make_single_trajectory_mc_visual(
                q_pred,
                mc_returns,
            )
        )

    return np.concatenate(visualization_images, 0)
