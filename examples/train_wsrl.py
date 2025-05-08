"""
zhouzypaul: script to run WSRL
modified from train_rlpd.py and instead do:
0. load in pre-trained Q/Pi
1. no data retention and no 50/50 sampling
2. warmup
"""

#!/usr/bin/env python3

import copy
import glob
import os
import pickle as pkl
import time

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from agentlace.data.data_store import QueuedDataStore
from agentlace.trainer import TrainerClient, TrainerServer
from experiments.mappings import CONFIG_MAPPING
from flax.training import checkpoints
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from natsort import natsorted
from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_dual_arm,
    make_sac_pixel_agent_hybrid_single_arm,
    make_sac_pixel_agent_with_resnet_mlp,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.train_utils import concat_batches
from serl_launcher.wrappers.past_n_statistic import RecordPastNStatisticsWrapper

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_string("description", "", "Wandb exp name")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("learner", False, "Whether this is a learner.")
flags.DEFINE_boolean("actor", False, "Whether this is an actor.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_multi_string("offline_data_path", None, "Path to the offline data.")
flags.DEFINE_string("save_path", None, "Path to save checkpoints.")
flags.DEFINE_integer("eval_checkpoint_step", 0, "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_boolean("save_video", False, "Save video.")
flags.DEFINE_boolean("use_resnet_mlp", False, "Use resnet mlp.")

# env
flags.DEFINE_float("reward_scale", 1.0, "Reward scale.")
flags.DEFINE_float("reward_bias", -1.0, "Reward bias. Default to step penalty rewards.")

# WSRL args
flags.DEFINE_string(
    "pretrained_checkpoint_path",
    None,
    "Path to the pre-trained offline RL agent checkpoint.",
)
flags.DEFINE_float(
    "offline_data_ratio", 0, "Ratio of offline data to sample in a batch. WSRL uses 0."
)
flags.DEFINE_integer("warmup_period", 5000, "Number of warmup steps for WSRL.")

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging


devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


##############################################################################


def actor(agent, data_store, intvn_data_store, env, sampling_rng):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    if FLAGS.eval_checkpoint_step:
        print_green(
            "Eval loop with checkpoint at step {}".format(FLAGS.eval_checkpoint_step)
        )
        success_counter = 0
        time_list = []

        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.save_path),
            agent.state,
            step=FLAGS.eval_checkpoint_step,
        )
        agent = agent.replace(state=ckpt)

        for episode in range(FLAGS.eval_n_trajs):
            obs, _ = env.reset()
            done = False
            start_time = time.time()
            while not done:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs), argmax=False, seed=key
                )
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
        return  # after done eval, return and exit

    start_step = (
        int(
            os.path.basename(
                natsorted(glob.glob(os.path.join(FLAGS.save_path, "buffer/*.pkl")))[-1]
            )[12:-4]
        )
        + 1
        if FLAGS.save_path and glob.glob(os.path.join(FLAGS.save_path, "checkpoint_*"))
        else 0
    )

    datastore_dict = {
        "actor_env": data_store,
        "actor_env_intvn": intvn_data_store,
    }

    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_stores=datastore_dict,
        wait_for_server=True,
        timeout_ms=3000,
    )

    # Function to update the agent with new params
    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

    transitions = []
    demo_transitions = []

    obs, _ = env.reset()
    done = False

    # training loop
    timer = Timer()
    running_return = 0.0
    already_intervened = False
    intervention_count = 0
    intervention_steps = 0

    pbar = tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True)
    for step in pbar:
        timer.tick("total")

        with timer.context("sample_actions"):
            if step < config.random_steps:
                actions = env.action_space.sample()
            else:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    seed=key,
                    argmax=False,
                )
                actions = np.asarray(jax.device_get(actions))

        # Step environment
        with timer.context("step_env"):

            next_obs, reward, done, truncated, info = env.step(actions)
            if "left" in info:
                info.pop("left")
            if "right" in info:
                info.pop("right")

            # override the action with the intervention action
            if "intervene_action" in info:
                actions = info.pop("intervene_action")
                intervention_steps += 1
                if not already_intervened:
                    intervention_count += 1
                already_intervened = True
            else:
                already_intervened = False

            running_return += reward
            transition = dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=reward,
                masks=1.0 - done,
                dones=done,
            )
            if "grasp_penalty" in info:
                transition["grasp_penalty"] = info["grasp_penalty"]
            data_store.insert(transition)
            transitions.append(copy.deepcopy(transition))
            if already_intervened:
                intvn_data_store.insert(transition)
                demo_transitions.append(copy.deepcopy(transition))

            obs = next_obs
            if done or truncated:
                info["episode"]["intervention_count"] = intervention_count
                info["episode"]["intervention_steps"] = intervention_steps
                stats = {"environment": info}  # send stats to the learner to log
                client.request("send-stats", stats)
                pbar.set_description(f"last return: {running_return}")
                running_return = 0.0
                intervention_count = 0
                intervention_steps = 0
                already_intervened = False
                client.update()
                obs, _ = env.reset()

        if step > 0 and config.buffer_period > 0 and step % config.buffer_period == 0:
            # dump to pickle file
            buffer_path = os.path.join(FLAGS.save_path, "buffer")
            demo_buffer_path = os.path.join(FLAGS.save_path, "demo_buffer")
            if not os.path.exists(buffer_path):
                os.makedirs(buffer_path)
            if not os.path.exists(demo_buffer_path):
                os.makedirs(demo_buffer_path)
            with open(os.path.join(buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                pkl.dump(transitions, f)
                transitions = []
            with open(
                os.path.join(demo_buffer_path, f"transitions_{step}.pkl"), "wb"
            ) as f:
                pkl.dump(demo_transitions, f)
                demo_transitions = []

        timer.tock("total")

        if step % config.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)


##############################################################################


def learner(rng, agent, replay_buffer, offline_buffer, wandb_logger=None):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    start_step = (
        int(
            os.path.basename(
                checkpoints.latest_checkpoint(os.path.abspath(FLAGS.save_path))
            )[11:]
        )
        + 1
        if FLAGS.save_path and glob.glob(os.path.join(FLAGS.save_path, "checkpoint_*"))
        else 0
    )
    step = start_step

    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            payload["actor_steps"] = len(replay_buffer)  # add in the actor steps
            wandb_logger.log(payload, step=step)
        return {}  # not expecting a response

    # Create server
    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    if offline_buffer:
        server.register_data_store("actor_env_intvn", offline_buffer)
    server.start(threaded=True)

    # Loop to wait until warmup steps is filled
    pbar = tqdm.tqdm(
        total=FLAGS.warmup_period,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < FLAGS.warmup_period:
        pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
    pbar.close()

    # send the initial network to the actor
    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    # sample the data from replay buffer and offline buffer
    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": int(config.batch_size * (1 - FLAGS.offline_data_ratio)),
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )
    if offline_buffer and FLAGS.offline_data_ratio > 0:
        offline_iterator = offline_buffer.get_iterator(
            sample_args={
                "batch_size": int(config.batch_size * FLAGS.offline_data_ratio),
                "pack_obs_and_next_obs": True,
            },
            device=sharding.replicate(),
        )
    else:
        offline_iterator = None

    # wait till the replay buffer is filled with enough data
    timer = Timer()

    if isinstance(agent, SACAgent):
        train_critic_networks_to_update = frozenset({"critic"})
        train_networks_to_update = frozenset({"critic", "actor", "temperature"})
    else:
        train_critic_networks_to_update = frozenset({"critic", "grasp_critic"})
        train_networks_to_update = frozenset(
            {"critic", "grasp_critic", "actor", "temperature"}
        )

    for step in tqdm.tqdm(
        range(start_step, config.max_steps), dynamic_ncols=True, desc="learner"
    ):
        # run n-1 critic updates and 1 critic + actor update.
        # This makes training on GPU faster by reducing the large batch transfer time from CPU to GPU
        for critic_step in range(config.cta_ratio - 1):
            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)
                if offline_iterator:
                    offline_batch = next(offline_iterator)
                    batch = concat_batches(batch, offline_batch, axis=0)

            with timer.context("train_critics"):
                agent, critics_info = agent.update(
                    batch,
                    networks_to_update=train_critic_networks_to_update,
                )

        with timer.context("train"):
            batch = next(replay_iterator)
            if offline_iterator:
                offline_batch = next(offline_iterator)
                batch = concat_batches(batch, offline_batch, axis=0)
            agent, update_info = agent.update(
                batch,
                networks_to_update=train_networks_to_update,
            )
        # publish the updated network
        if step > 0 and step % (config.steps_per_update) == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        if step % config.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=step)
            wandb_logger.log({"timer": timer.get_average_times()}, step=step)

        if config.checkpoint_period and (step + 1) % config.checkpoint_period == 0:
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.save_path), agent.state, step=step + 1, keep=100
            )


##############################################################################


def main(_):
    global config
    config = CONFIG_MAPPING[FLAGS.exp_name]()

    assert config.batch_size % num_devices == 0
    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    env = config.get_environment(
        fake_env=FLAGS.learner,
        save_video=FLAGS.save_video,
        classifier=True,
    )
    env = RecordEpisodeStatistics(env)
    env = RecordPastNStatisticsWrapper(env, n=10)

    rng, sampling_rng = jax.random.split(rng)

    if (
        config.setup_mode == "single-arm-fixed-gripper"
        or config.setup_mode == "dual-arm-fixed-gripper"
    ):
        agenttype = make_sac_pixel_agent_with_resnet_mlp if FLAGS.use_resnet_mlp else make_sac_pixel_agent
        agent: SACAgent = agenttype(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
            reward_scale=FLAGS.reward_scale,
            reward_bias=FLAGS.reward_bias,
        )
        include_grasp_penalty = False
    elif config.setup_mode == "single-arm-learned-gripper":
        agent: SACAgentHybridSingleArm = make_sac_pixel_agent_hybrid_single_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = True
    elif config.setup_mode == "dual-arm-learned-gripper":
        agent: SACAgentHybridDualArm = make_sac_pixel_agent_hybrid_dual_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = True
    else:
        raise NotImplementedError(f"Unknown setup mode: {config.setup_mode}")

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent = jax.device_put(jax.tree_map(jnp.array, agent), sharding.replicate())

    # load from the pre-trained offline RL agent
    assert FLAGS.pretrained_checkpoint_path is not None
    ckpt = checkpoints.restore_checkpoint(
        os.path.abspath(FLAGS.pretrained_checkpoint_path),
        agent.state,
    )
    agent = agent.replace(state=ckpt)
    print_green(
        f"Loaded pre-trained checkpoint from {FLAGS.pretrained_checkpoint_path}"
    )

    # resume from a previous run
    if (
        FLAGS.save_path is not None
        and os.path.exists(FLAGS.save_path)
        and glob.glob(os.path.join(FLAGS.save_path, "checkpoint_*"))
    ):
        input(
            "Checkpoint path already exists from a previous run. Press Enter to resume training."
        )
        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.save_path),
            agent.state,
        )
        agent = agent.replace(state=ckpt)
        ckpt_number = os.path.basename(
            checkpoints.latest_checkpoint(os.path.abspath(FLAGS.save_path))
        )[11:]
        print_green(f"Loaded previous checkpoint at step {ckpt_number}. Resuming ...")

    def create_replay_buffer_and_wandb_logger():
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
        )
        # set up wandb and logging
        wandb_logger = make_wandb_logger(
            project="hil-serl",
            description=f"wsrl_{FLAGS.description}_from_{FLAGS.pretrained_checkpoint_path.split('/')[-2]}",
            debug=FLAGS.debug,
            variant={
                **FLAGS.flag_values_dict(),
                "agent_config": dict(**agent.config),
            },
        )
        return replay_buffer, wandb_logger

    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        replay_buffer, wandb_logger = create_replay_buffer_and_wandb_logger()
        print_green(f"Online buffer size: {len(replay_buffer)}")

        # create offline buffer if needed
        offline_buffer = None
        if FLAGS.offline_data_path is not None:
            offline_buffer = MemoryEfficientReplayBufferDataStore(
                env.observation_space,
                env.action_space,
                capacity=config.replay_buffer_capacity,
                image_keys=config.image_keys,
                include_grasp_penalty=include_grasp_penalty,
            )

            for path in FLAGS.offline_data_path:
                with open(path, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        if (
                            "infos" in transition
                            and "grasp_penalty" in transition["infos"]
                        ):
                            transition["grasp_penalty"] = transition["infos"][
                                "grasp_penalty"
                            ]
                        offline_buffer.insert(transition)
            print_green(f"Offline buffer size: {len(offline_buffer)}")

        # continue from a previous WSRL run's data
        if FLAGS.save_path is not None and os.path.exists(
            os.path.join(FLAGS.save_path, "buffer")
        ):
            for file in glob.glob(os.path.join(FLAGS.save_path, "buffer/*.pkl")):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        replay_buffer.insert(transition)
            print_green(
                f"Loaded previous buffer data. Replay buffer size: {len(replay_buffer)}"
            )

        # if offline_buffer and FLAGS.save_path is not None and os.path.exists(
        #     os.path.join(FLAGS.save_path, "demo_buffer")
        # ):
        #     for file in glob.glob(
        #         os.path.join(FLAGS.save_path, "demo_buffer/*.pkl")
        #     ):
        #         with open(file, "rb") as f:
        #             transitions = pkl.load(f)
        #             for transition in transitions:
        #                 offline_buffer.insert(transition)
        #     print_green(
        #         f"Loaded previous demo buffer data. Demo buffer size: {len(offline_buffer)}"
        #     )

        # learner loop
        print_green("starting learner loop")
        learner(
            sampling_rng,
            agent,
            replay_buffer,
            offline_buffer=offline_buffer,
            wandb_logger=wandb_logger,
        )

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store = QueuedDataStore(50000)  # the queue size on the actor
        intvn_data_store = QueuedDataStore(50000)

        # actor loop
        print_green("starting actor loop")
        actor(
            agent,
            data_store,
            intvn_data_store,
            env,
            sampling_rng,
        )

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)
