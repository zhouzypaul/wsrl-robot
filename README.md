# WSRL: Efficient Online Reinforcement Learning Fine-Tuning Need Not Retain Offline Data

This project is built on top of HIL-SERL. See their README below for more details.

## Installation
See [Installation](#installation) and follow the steps to set up your environment

## File Structure

 The SERL, CalQL, and WSRL training files are in `/examples`.`examples/experiments/peg_insertion/config.py` contains our reset poses and training hyperparameters. See their [code structure](#overview-and-code-structure) for more details.
## Running WSRL on Franka Peg Insertion

1. **Setup**

    ```
    cd examples
    ```

    Train the reward classifier and collect 20 expert demos. For the reward classifier we define anything from a half-insert to a full insert as success and collect many near-inserts as failures for robustness.

    ```
    python train_reward_classifier.py --exp_name peg_insertion
    ```

    ```
    python record_demos.py --exp_name peg_insertion --successes_needed 20
    ```
    Follow the [franka_walkthrough](./docs/franka_walkthrough.md) RAM insertion steps for more info. Note that our exp_name is peg_insertion and our corresponding files are in `examples/experiments/peg_insertion`

2. **Collecting Offline Data**

    Use SERL or HIL-SERL to collect a dataset of robot transitions. We find that giving ~10 interventions near the start of training works best for a dataset of ~20k transitions.

    Note to include your expert demo path in `experiments/peg_insertion/run_actor.sh` first.

    ```
    sh experiments/peg_insertion/run_actor.sh --checkpoint_path [logs/offline_data_path]--description [hil_serl_data_collection]--use_resnet_mlp
    ```
    ```
    sh experiments/peg_insertion/run_learner.sh --checkpoint_path [logs/offline_data_path] --description [hil_serl_data_collection]--use_resnet_mlp
    ```

3. **Training offline Calql**

    Run this script to train offline CalQL and save checkpoints to `calql_checkpoint_path` using your offline dataset from `offline_data_path`. We found that training for ~200k steps converges and achieves 13/20 performance on our evals.
    ```
    bash experiments/peg_insertion/run_calql_pretrain.sh --calql_checkpoint_path [logs/calql_checkpoint_path] --data_path [logs/offline_data_path/buffer] --use_resnet_mlp
    ```
    Use the `--eval_n_trajs` flag to evaluate your CalQL checkpoint.
    ```
    bash experiments/peg_insertion/run_calql_pretrain.sh --calql_checkpoint_path [logs/calql_checkpoint_path] --eval_n_trajs 20 --use_resnet_mlp
    ```

4. **Running WSRL online**

    Initialize WSRL with your pretrained CalQL checkpoint, and run the training script.
    ```
    sh experiments/peg_insertion/run_wsrl_actor.sh --save_path [logs/wsrl_save_path] --description [wsrl_run]--use_resnet_mlp
    ```
    ```
    sh experiments/peg_insertion/run_wsrl_learner.sh --save_path [logs/wsrl_save_path] --description [wsrl_run]--use_resnet_mlp
    ```

    Evaluate your WSRL checkpoint
    ```
    sh experiments/peg_insertion/run_wsrl_eval.sh --use_resnet_mlp
    ```


# HIL-SERL: Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Static Badge](https://img.shields.io/badge/Project-Page-a)](https://hil-serl.github.io/)
[![Discord](https://img.shields.io/discord/1302866684612444190?label=Join%20Us%20on%20Discord&logo=discord&color=7289da)](https://discord.gg/G4xPJEhwuC)


![](./docs/images/task_banner.gif)


**Webpage: [https://hil-serl.github.io/](https://hil-serl.github.io/)**

HIL-SERL provides a set of libraries, env wrappers, and examples to train RL policies using a combination of demonstrations and human corrections to perform robotic manipulation tasks with near-perfect success rates. The following sections describe how to use HIL-SERL. We will illustrate the usage with examples.

🎬: [HIL-SERL video](https://www.youtube.com/watch?v=GuD_-zhJgbs)

**Table of Contents**
- [HIL-SERL: Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning](#serl-a-software-suite-for-sample-efficient-robotic-reinforcement-learning)
  - [Installation](#installation)
  - [Overview and Code Structure](#overview-and-code-structure)
  - [Run with Franka Arm](#run-with-franka-arm)
  <!-- - [Contribution](#contribution) -->
  - [Citation](#citation)

## Installation
1. **Setup Conda Environment:**
    create an environment with
    ```bash
    conda create -n hilserl python=3.10
    ```

2. **Install Jax as follows:**
    - For CPU (not recommended):
        ```bash
        pip install --upgrade "jax[cpu]"
        ```

    - For GPU:
        ```bash
        pip install --upgrade "jax[cuda12_pip]==0.4.35" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
        ```

    - For TPU
        ```bash
        pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
        ```
    - See the [Jax Github page](https://github.com/google/jax) for more details on installing Jax.

3. **Install the serl_launcher**
    ```bash
    cd serl_launcher
    pip install -e .
    pip install -r requirements.txt
    ```

4. **Install for serl_robot_infra** Follow the [README](./serl_robot_infra/README.md) in `serl_robot_infra` for installation and basic robot operation instructions. This contains the instruction for installing the impendence-based [serl_franka_controllers](https://github.com/rail-berkeley/serl_franka_controllers). After the installation, you should be able to run the robot server, interact with the gym `franka_env` (hardware).

## Overview and Code Structure

HIL-SERL provides a set of common libraries for users to train RL policies for robotic manipulation tasks. The main structure of running the RL experiments involves having an actor node and a learner node, both of which interact with the robot gym environment. Both nodes run asynchronously, with data being sent from the actor to the learner node via the network using [agentlace](https://github.com/youliangtan/agentlace). The learner will periodically synchronize the policy with the actor. This design provides flexibility for parallel training and inference.

<!-- <p align="center">
  <img src="./docs/images/software_design.png" width="80%"/>
</p> -->

**Table for code structure**

| Code Directory | Description |
| --- | --- |
| [examples](https://github.com/rail-berkeley/hil-serl/blob/main/examples) | Scripts for policy training, demonstration data collection, reward classifier training |
| [serl_launcher](https://github.com/rail-berkeley/hil-serl/blob/main/serl_launcher) | Main code for HIL-SERL |
| [serl_launcher.agents](https://github.com/rail-berkeley/hil-serl/blob/main/serl_launcher/serl_launcher/agents/) | Agent Policies (e.g. SAC, BC) |
| [serl_launcher.wrappers](https://github.com/rail-berkeley/hil-serl/blob/main/serl_launcher/serl_launcher/wrappers) | Gym env wrappers |
| [serl_launcher.data](https://github.com/rail-berkeley/hil-serl/blob/main/serl_launcher/serl_launcher/data) | Replay buffer and data store |
| [serl_launcher.vision](https://github.com/rail-berkeley/hil-serl/blob/main/serl_launcher/serl_launcher/vision) | Vision related models and utils |
| [serl_robot_infra](./serl_robot_infra/) | Robot infra for running with real robots |
| [serl_robot_infra.robot_servers](https://github.com/rail-berkeley/hil-serl/blob/main/serl_robot_infra/robot_servers/) | Flask server for sending commands to robot via ROS |
| [serl_robot_infra.franka_env](https://github.com/rail-berkeley/hil-serl/blob/main/serl_robot_infra/franka_env/) | Gym env for Franka robot |

## Run with Franka Arm

We provide a step-by-step guide to run RL policies with HIL-SERL on a Franka robot.

Check out the [Run with Franka Arm](/docs/franka_walkthrough.md)
 - [RAM Insertion](/docs/franka_walkthrough.md#1-ram-insertion)
 - [USB Pickup and Insertion](/docs/franka_walkthrough.md#2-usb-pick-up-and-insertion)
 - [Object Handover](/docs/franka_walkthrough.md#3-object-handover)
 - [Egg Flip](/docs/franka_walkthrough.md#4-egg-flip)

<!-- ## Contribution

We welcome contributions to this repository! Fork and submit a PR if you have any improvements to the codebase. Before submitting a PR, please run `pre-commit run --all-files` to ensure that the codebase is formatted correctly. -->

## Citation

If you use this code for your research, please cite our paper:

```bibtex
@misc{luo2024hilserl,
      title={Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning},
      author={Jianlan Luo and Charles Xu and Jeffrey Wu and Sergey Levine},
      year={2024},
      eprint={2410.21845},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```
