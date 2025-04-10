#!/usr/bin/env python3

import glob
import os
import pickle as pkl
import time
from typing import Union

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from experiments.configs.train_config import DefaultTrainingConfig
from experiments.mappings import CONFIG_MAPPING
from flax.training import checkpoints
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from serl_launcher.agents.continuous.calql import CalQLAgent
from serl_launcher.agents.continuous.cql import CQLAgent
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
from serl_launcher.utils.launcher import make_calql_pixel_agent, make_wandb_logger

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_string("group", None, "Name of wandb group")
flags.DEFINE_string("description", "", "Wandb exp name")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_string("calql_checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_integer(
    "checkpoint_step", None, "Checkpoint step. If None, use the latest checkpoint."
)
flags.DEFINE_integer(
    "save_period", 50_000, "How often to save checkpoints during training."
)
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_integer("train_steps", 200_000, "Number of pretraining steps.")
flags.DEFINE_bool("save_video", False, "Save video of the evaluation.")
flags.DEFINE_string("data_path", None, "Path to the demo data.")
flags.DEFINE_bool("use_calql", True, "Use CalQL instead of CQL.")
flags.DEFINE_float("reward_scale", 1.0, "Reward scale")
flags.DEFINE_float("reward_bias", 0.0, "Reward bias")


flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging


devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


def print_yellow(x):
    return print("\033[93m {}\033[00m".format(x))


##############################################################################


def eval(
    env,
    calql_agent: CalQLAgent,
    sampling_rng,
):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    success_counter = 0
    time_list = []
    for episode in range(FLAGS.eval_n_trajs):
        obs, _ = env.reset()
        done = False
        start_time = time.time()
        while not done:
            rng, key = jax.random.split(sampling_rng)

            actions = calql_agent.sample_actions(observations=obs, seed=key)
            actions = np.asarray(jax.device_get(actions))
            next_obs, reward, done, truncated, info = env.step(actions)
            obs = next_obs
            if done:
                if reward:
                    dt = time.time() - start_time
                    time_list.append(dt)
                    print(dt)
                success_counter += reward
                print(reward)
                print(f"{success_counter}/{episode + 1}")

    print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
    print(f"average time: {np.mean(time_list)}")


##############################################################################


def train(
    calql_agent: CalQLAgent,
    demo_buffer,
    config: DefaultTrainingConfig,
    wandb_logger=None,
):

    calql_replay_iterator = demo_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size,
            "pack_obs_and_next_obs": False,
        },
        device=sharding.replicate(),
    )

    # Pretrain CalQL policy to get started
    for step in tqdm.tqdm(
        range(FLAGS.train_steps),
        dynamic_ncols=True,
        desc="calql_pretraining",
    ):
        batch = next(calql_replay_iterator)
        calql_agent, calql_update_info = calql_agent.update(batch)
        if step % config.log_period == 0 and wandb_logger:
            wandb_logger.log({"calql": calql_update_info}, step=step)

        if FLAGS.save_period and (step + 1) % FLAGS.save_period == 0:
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.calql_checkpoint_path),
                calql_agent.state,
                step=step + 1,
                keep=100,
            )

    print_green("calql pretraining done and saved checkpoint")


##############################################################################


def main(_):
    config: DefaultTrainingConfig = CONFIG_MAPPING[FLAGS.exp_name]()

    assert config.batch_size % num_devices == 0
    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    eval_mode = FLAGS.eval_n_trajs > 0
    env = config.get_environment(
        fake_env=not eval_mode,
        save_video=FLAGS.save_video,
        classifier=True,
    )
    env = RecordEpisodeStatistics(env)

    calql_agent: Union[CalQLAgent, CQLAgent] = make_calql_pixel_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=config.image_keys,
        encoder_type=config.encoder_type,
        reward_scale=FLAGS.reward_scale,
        reward_bias=FLAGS.reward_bias,  # eventually move this to the env?
        discount=config.discount,
        is_calql=FLAGS.use_calql,
    )

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    calql_agent: CalQLAgent = jax.device_put(
        jax.tree_map(jnp.array, calql_agent), sharding.replicate()
    )

    if not eval_mode:
        assert not os.path.isdir(
            os.path.join(FLAGS.calql_checkpoint_path, f"checkpoint_{FLAGS.train_steps}")
        )

        demo_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_mc_returns=FLAGS.use_calql,
            discount=config.discount,
            reward_scale=FLAGS.reward_scale,
            reward_bias=FLAGS.reward_bias,
        )

        # set up wandb and logging
        wandb_logger = make_wandb_logger(
            project="hil-serl",
            description=f"calql_alpha_{calql_agent.config['cql_alpha']}_reward_bias_{FLAGS.reward_bias}_discount_{config.discount}_{FLAGS.description}",
            debug=FLAGS.debug,
            group=FLAGS.group,
            variant={
                **FLAGS.flag_values_dict(),
                "agent_config": dict(**calql_agent.config),
            },
            entity="rl-finetune",
        )

        assert FLAGS.data_path is not None
        assert os.path.isdir(FLAGS.data_path)

        for path in glob.glob(os.path.join(FLAGS.data_path, "*.pkl")):
            with open(path, "rb") as f:
                print_green(f"Loading {path}")
                transitions = pkl.load(f)
                for transition in transitions:
                    demo_buffer.insert(transition)
        print_green(f"demo buffer size: {len(demo_buffer)}")
        # learner loop
        print_green("starting learner loop")
        train(
            calql_agent=calql_agent,
            demo_buffer=demo_buffer,
            wandb_logger=wandb_logger,
            config=config,
        )

    else:
        rng = jax.random.PRNGKey(FLAGS.seed)
        sampling_rng = jax.device_put(rng, sharding.replicate())

        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.calql_checkpoint_path),
            calql_agent.state,
            step=FLAGS.checkpoint_step,
        )
        calql_agent = calql_agent.replace(state=ckpt)

        print_green("starting actor loop")
        eval(
            env=env,
            calql_agent=calql_agent,
            sampling_rng=sampling_rng,
        )


if __name__ == "__main__":
    app.run(main)
