import os
import pickle as pkl
from typing import Union

import jax
import numpy as np
import wandb
from absl import app, flags
from experiments.configs.train_config import DefaultTrainingConfig
from experiments.mappings import CONFIG_MAPPING
from serl_launcher.agents.continuous.calql import CalQLAgent
from serl_launcher.agents.continuous.cql import CQLAgent
from serl_launcher.common.visualization import mc_q_visualization
from serl_launcher.utils.launcher import make_calql_pixel_agent, make_wandb_logger

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Experiment name")
flags.DEFINE_string("calql_checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_string("data_path", None, "Path to the demo data.")
flags.DEFINE_string("description", None, "Description of the experiment.")
flags.DEFINE_bool("use_calql", True, "Use CalQL instead of CQL.")
flags.DEFINE_float("reward_scale", 1.0, "Reward scale")
flags.DEFINE_float("reward_bias", 0.0, "Reward bias")
flags.DEFINE_integer("seed", 42, "Random seed.")


devices = jax.local_devices()
num_devices = len(devices)


def _batch_dicts(stat):
    """stat is a list of dict, turn it into a dict of list"""
    d = {}
    for k in stat[0].keys():
        d[k] = np.array([s[k] for s in stat])
    return d


def main(_):
    config: DefaultTrainingConfig = CONFIG_MAPPING[FLAGS.exp_name]()

    assert config.batch_size % num_devices == 0
    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    assert FLAGS.calql_checkpoint_path is not None and os.path.isdir(
        FLAGS.calql_checkpoint_path
    ), "Invalid checkpoint path."

    env = config.get_environment(
        fake_env=True,
        save_video=False,
        classifier=True,
    )

    calql_agent: Union[CalQLAgent, CQLAgent] = make_calql_pixel_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=config.image_keys,
        encoder_type=config.encoder_type,
        reward_scale=FLAGS.reward_scale,
        reward_bias=FLAGS.reward_bias,
        is_calql=FLAGS.use_calql,
    )

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    calql_agent: CalQLAgent = jax.device_put(
        jax.tree_map(jnp.array, calql_agent), sharding.replicate()
    )

    calql_ckpt = checkpoints.restore_checkpoint(
        os.path.abspath(FLAGS.calql_checkpoint_path),
        calql_agent.state,
    )
    calql_agent = calql_agent.replace(state=calql_ckpt)

    wandb_logger = make_wandb_logger(
        project="hil-serl",
        description=FLAGS.description,
        variant={
            "agent_config": calql_agent.config,
        },
    )

    # get trajectories
    assert FLAGS.data_path is not None and os.path.isfile(
        FLAGS.data_path
    ), "Invalid data path."
    trajectories = []
    with open(FLAGS.data_path, "rb") as f:
        transitions = pkl.load(f)
        trajectory = []
        for transition in transitions:
            trajectory.append(transition)
            if transition["dones"]:
                trajectories.append(_batch_dicts(trajectory))
                trajectory = []

    print_green(f"loaded {len(trajectories)} trajectories")
    wandb_logger.log(
        {
            "evaluation_visualization": wandb.Image(
                mc_q_visualization(trajectories, calql_agent, exp_name=FLAGS.exp_name)
            )
        },
    )


if __name__ == "__main__":
    app.run(main)
