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
            0.6667957108097786,
            -0.20343671720464337,
            0.11058320865610131,
            3.113260147874013,
            -0.0028925327830766623,
            1.5886509497093098,
        ]
    )  # peg fully inserted

    # GRASP_POSE = np.array([0.5857508505445138,-0.22036261105675414,0.2731021902359492, np.pi, 0, 0])  # when grasping peg sitting on the holder
    RESET_POSE = TARGET_POSE + np.array(
        [0, 0, 0.1, 0, 0, 0]
    )  # where the arm should reset to

    # randomness in reset
    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.05  # 0.1

    RANDOM_X_RANGE = (0.05, 0.05)
    RANDOM_Y_RANGE = (0.05, 0.05)

    RANDOM_RZ_RANGE = np.pi / 6
    DETERMINISTIC_RESET = True
    RESET_POSITIONS = [
        [
            0.699889649029879,
            -0.13460473617960322,
            0.18852075918891614,
            3.0590175760041918,
            0.06258653132579828,
            1.646078716944964,
        ],
        [
            0.7100925108472831,
            -0.17475509446847534,
            0.1878593167027761,
            3.0442761135190777,
            0.03834302920394883,
            1.6351021080636317,
        ],
        [
            0.7159087440108571,
            -0.22413363709827558,
            0.19256217625655148,
            3.1073435681759336,
            0.07364704735441108,
            1.6396906528944046,
        ],
        [
            0.7140523612972279,
            -0.2545531308303405,
            0.19284608054384705,
            3.1091382428133643,
            0.07112117039524546,
            1.6011069596694427,
        ],
        [
            0.6772899069734942,
            -0.1317739852417371,
            0.17927446567145872,
            -3.114299339045671,
            0.09076810614854636,
            1.5785690827897088,
        ],
        [
            0.6715418118336924,
            -0.16968458490018767,
            0.17813954994860381,
            -3.1231507506163307,
            0.07342815374666811,
            1.6065423159315146,
        ],
        [
            0.6793471109746927,
            -0.2000161583740931,
            0.1838967699428107,
            3.1382582719157868,
            0.06563555962059753,
            1.541816567250708,
        ],
        [
            0.6749983638923734,
            -0.23367025228172567,
            0.17916880097432314,
            3.1339207510166083,
            0.04910663780575053,
            1.5951285930742949,
        ],
        [
            0.6713488749105098,
            -0.26544406384445357,
            0.18015566986267062,
            3.1266789612114185,
            0.04159020524060897,
            1.5894353953282825,
        ],
        [
            0.6327902606304047,
            -0.14373186790211073,
            0.16495669544355612,
            3.131859043166398,
            0.0020782108040764413,
            1.5827869573180615,
        ],
        [
            0.6229155741818949,
            -0.177490120624203,
            0.16421760045389144,
            -3.138633644990577,
            0.015447299967091821,
            1.5814034272999102,
        ],
        [
            0.6132384021402353,
            -0.21027499103950198,
            0.1646751676023659,
            -3.1389672199773813,
            0.014813803673298098,
            1.5761978398590255,
        ],
        [
            0.6136665134637046,
            -0.24388819041100301,
            0.1711201924488995,
            3.131357147302584,
            0.018911430186261624,
            1.559346869268838,
        ],
        [
            0.6141438031298478,
            -0.27213505091437135,
            0.17778655970309465,
            3.117445049030777,
            0.022942387174223144,
            1.552481845499294,
        ],
        [
            0.624500505340127,
            -0.23889273702583735,
            0.20943305760891287,
            3.0915221604486804,
            0.031928655333153255,
            0.9116579724982989,
        ],
        [
            0.6241733100281689,
            -0.16585782263425972,
            0.1914736451795393,
            -3.122902066439545,
            0.018264666702152876,
            2.2124897308265132,
        ],
        [
            0.7086240249259236,
            -0.16986743981653646,
            0.20527548862246242,
            3.0686558971870195,
            0.033375844949012246,
            1.133171207442233,
        ],
        [
            0.7128933918608176,
            -0.24685604475407155,
            0.1904994397115377,
            3.133965801520982,
            0.036881030846101615,
            1.9491467830700373,
        ],
        [
            0.6868646589674893,
            -0.19834540482969898,
            0.20222559974216964,
            -3.112787730950409,
            0.05980821801785341,
            2.313122208693121,
        ],
        [
            0.6838755887755026,
            -0.2062957678132729,
            0.17444994874201886,
            -3.1355200891562354,
            0.049855713565423176,
            1.591024396885016,
        ],
    ]  # reset positions for deterministic reset

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
        "translational_clip_neg_x": 0.008,
        "translational_clip_neg_y": 0.003,
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
    checkpoint_period = 500
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
