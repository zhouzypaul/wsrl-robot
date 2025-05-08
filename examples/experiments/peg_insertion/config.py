import os

import jax
import jax.numpy as jnp
import numpy as np
from experiments.configs.train_config import DefaultTrainingConfig
from experiments.peg_insertion.wrapper import PegEnv
from franka_env.envs.franka_env import DefaultEnvConfig
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.wrappers import (
    GripperCloseEnv,
    MultiCameraBinaryRewardClassifierWrapper,
    Quat2EulerWrapper,
    SpacemouseIntervention,
)
from serl_launcher.networks.reward_classifier import load_classifier_func
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper


class EnvConfig(DefaultEnvConfig):
    SERVER_URL = "http://127.0.0.1:5000/"
    REALSENSE_CAMERAS = {
        "wrist_1": {
            "serial_number": "128422272758",
            "dim": (1280, 720),
            "exposure": 40000,
        },
        "wrist_2": {
            "serial_number": "127122270572",
            "dim": (1280, 720),
            "exposure": 40000,
        },
    }
    IMAGE_CROP = {
        # "wrist_1": lambda img: img[
        #     150:450, 350:1100
        # ],
        # "wrist_2": lambda img: img[
        #     100:500, 400:900
        # ],  # zhouzypaul: I removed image crop and use full obs
    }
    DISPLAY_IMAGE = True

    TARGET_POSE = np.array(
        [
            0.5781350843740655,
            -0.07251673541653851,
            0.10484627189030093,
            3.090419581246303,
            0.02255417062889542,
            1.610634991219969,
        ]
    )  # peg fully inserted
    # GRASP_POSE = np.array([0.5857508505445138,-0.22036261105675414,0.2731021902359492, np.pi, 0, 0])  # when grasping peg sitting on the holder
    RESET_POSE = TARGET_POSE + np.array(
        [0, 0, 0.1, 0, 0, 0]
    )  # where the arm should reset to

    # randomness in reset
    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.05  # 0.1
    RANDOM_X_RANGE = (0.05, 0.2)
    RANDOM_Y_RANGE = (0.1, 0.1)

    # RANDOM_X_RANGE = (0.05, 0.05)
    # RANDOM_Y_RANGE = (0.05, 0.05)
    RANDOM_RZ_RANGE = np.pi / 6
    DETERMINISTIC_RESET = False
    RESET_POSITIONS = [
        [
            0.5456584536113785,
            -0.13540999718100116,
            0.20428608510794843,
            -3.098707381676255,
            0.07922149617826024,
            1.6388225801839449,
        ],
        [
            0.5270387100675845,
            -0.07449608843946603,
            0.19890491771315688,
            -3.062442781416739,
            0.07320351467260289,
            1.6556212600885043,
        ],
        [
            0.5451024879461983,
            -0.019201971772620244,
            0.19445697757785313,
            -3.058801548151889,
            0.04131197907031825,
            1.683112826348169,
        ],
        [
            0.5880681483280971,
            -0.027274589454628018,
            0.1865156568480808,
            -3.1169355472443296,
            -0.0010051321363000465,
            1.6671977548860166,
        ],
        [
            0.6209454533132557,
            -0.025581233339754525,
            0.1844836233629435,
            -3.112781179804129,
            -0.038723291331560716,
            1.6614325781223596,
        ],
        [
            0.6244682223421955,
            -0.05663737942650188,
            0.19044701426542834,
            -3.128081815540121,
            -0.03629202195891712,
            1.643665185779923,
        ],
        [
            0.6214064736537981,
            -0.0997769062882742,
            0.19248708627403568,
            -3.134984288837611,
            -0.033196198872785576,
            1.6236498183500216,
        ],
        [
            0.622610629712021,
            -0.1387006679081746,
            0.18224002165501807,
            -3.1377464065094265,
            -0.05209244781779976,
            1.613556878584602,
        ],
        [
            0.589144451725303,
            -0.12452476686013347,
            0.18695272086612205,
            -3.1160473271355755,
            0.046793885015534054,
            1.6959241135229297,
        ],
        [
            0.5850827847764313,
            -0.08494523813316905,
            0.19158915085540124,
            -3.0943187012858555,
            0.0327602236977067,
            1.6315804231583322,
        ],
    ]

    # bounding box for the pos
    ABS_POSE_LIMIT_LOW = np.array(
        [
            TARGET_POSE[0] - RANDOM_X_RANGE[0],
            TARGET_POSE[1] - RANDOM_Y_RANGE[0],
            TARGET_POSE[2],
            TARGET_POSE[3] - 0.01,
            TARGET_POSE[4] - 0.01,
            TARGET_POSE[5] - RANDOM_RZ_RANGE,
        ]
    )
    ABS_POSE_LIMIT_HIGH = np.array(
        [
            TARGET_POSE[0] + RANDOM_X_RANGE[1],
            TARGET_POSE[1] + RANDOM_Y_RANGE[1],
            TARGET_POSE[2] + 0.1,
            TARGET_POSE[3] + 0.01,
            TARGET_POSE[4] + 0.01,
            TARGET_POSE[5] + RANDOM_RZ_RANGE,
        ]
    )

    ACTION_SCALE = (0.02, 0.1, 1)  # (xyz, r_xyz, gripper)
    MAX_EPISODE_LENGTH = 500  # TODO: 100 --> 300
    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "translational_clip_x": 0.008,
        "translational_clip_y": 0.003,
        "translational_clip_z": 0.004,
        "translational_clip_neg_x": 0.005,
        "translational_clip_neg_y": 0.002,
        "translational_clip_neg_z": 0.004,
        "rotational_clip_x": 0.02,
        "rotational_clip_y": 0.02,
        "rotational_clip_z": 0.02,
        "rotational_clip_neg_x": 0.02,
        "rotational_clip_neg_y": 0.02,
        "rotational_clip_neg_z": 0.02,
        "rotational_Ki": 0,
    }
    PRECISION_PARAM = {
        "translational_stiffness": 3000,
        "translational_damping": 89,
        "rotational_stiffness": 300,
        "rotational_damping": 9,
        "translational_Ki": 0.1,
        "translational_clip_x": 0.01,
        "translational_clip_y": 0.01,
        "translational_clip_z": 0.01,
        "translational_clip_neg_x": 0.01,
        "translational_clip_neg_y": 0.01,
        "translational_clip_neg_z": 0.01,
        "rotational_clip_x": 0.05,
        "rotational_clip_y": 0.05,
        "rotational_clip_z": 0.05,
        "rotational_clip_neg_x": 0.05,
        "rotational_clip_neg_y": 0.05,
        "rotational_clip_neg_z": 0.05,
        "rotational_Ki": 0.1,
    }

    # other original config from SERL code
    # REWARD_THRESHOLD: np.ndarray = np.array([0.01, 0.01, 0.01, 0.2, 0.2, 0.2])
    # ACTION_SCALE = (0.02, 0.1, 1)
    # RANDOM_XY_RANGE = 0.05
    # RESET_POSE = TARGET_POSE + np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0])
    # APPLY_GRIPPER_PENALTY = False
    # REWARD_THRESHOLD: np.ndarray = np.array([0.01, 0.01, 0.01, 0.2, 0.2, 0.2])


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["wrist_1", "wrist_2"]
    classifier_keys = ["wrist_1", "wrist_2"]
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    buffer_period = 1000
    checkpoint_period = 5000
    steps_per_update = 50
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-fixed-gripper"
    discount = 0.98
    batch_size = 256
    max_steps = 20000

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env = PegEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=EnvConfig(),
        )
        env = GripperCloseEnv(env)
        if not fake_env:
            env = SpacemouseIntervention(env)
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        if classifier:
            classifier = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                # (zhouzypaul: removed) added check for z position to further robustify classifier, but should work without as well
                prob = sigmoid(classifier(obs))[0]  # 0 to get the item from array
                return int(prob > 0.85)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        return env
