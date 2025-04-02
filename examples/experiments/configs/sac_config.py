import jax.nn as nn
from ml_collections import ConfigDict


def get_config(updates=None):
    config = ConfigDict()
    config.discount = 0.97
    config.backup_entropy = False
    config.target_entropy = None
    config.soft_target_update_rate = 0.005
    config.critic_ensemble_size = 10
    config.critic_subsample_size = 2
    config.autotune_entropy = True
    config.temperature_init = 1e-2

    # arch
    config.critic_network_kwargs = ConfigDict(
        {
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256, 256, 256],
        }
    )
    config.policy_network_kwargs = ConfigDict(
        {
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256, 256, 256],
        }
    )
    config.policy_kwargs = ConfigDict(
        {
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        }
    )

    config.actor_optimizer_kwargs = ConfigDict(
        {
            "learning_rate": 3e-4,
        }
    )
    config.critic_optimizer_kwargs = ConfigDict(
        {
            "learning_rate": 3e-4,
        }
    )
    config.temperature_optimizer_kwargs = ConfigDict(
        {
            "learning_rate": 3e-4,
        }
    )

    if updates is not None:
        config.update(ConfigDict(updates).copy_and_resolve_references())

    return config
