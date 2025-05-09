from typing import NamedTuple
import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from functools import partial


class ActorEqx(eqx.Module):
    layers: list

    def __init__(self, key, obs_size, action_size):
        keys = jax.random.split(key, 3)
        self.layers = [
            eqx.nn.Linear(obs_size, 64, key=keys[0]),
            eqx.nn.Linear(64, 64, key=keys[1]),
            eqx.nn.Linear(64, action_size, key=keys[2]),
        ]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x)


class TrainingStateEqxFlatten(NamedTuple):
    leaf_values: list
    opt_state: optax.OptState
    rng: jax.random.PRNGKey
    env_state: any
    obs: jnp.ndarray


@partial(jax.jit, static_argnums=(1))
def forward_pass_eqx_flatten(state: TrainingStateEqxFlatten, treedef):
    actor = jax.tree.unflatten(treedef, state.leaf_values)
    return actor(state.obs)


@partial(jax.jit, static_argnums=(1, 2, 3))
def collect_experience_eqx_flatten(state, env, env_params, treedef):
    leaf_values, opt_state, rng, env_state, obs = state
    rng, key_act, key_step = jax.random.split(rng, 3)

    actor = jax.tree.unflatten(treedef, leaf_values)
    logits = actor(obs)

    action = jax.random.categorical(key_act, logits)

    next_obs, next_env_state, reward, done, _ = env.step(
        key_step, env_state, action, env_params
    )

    # leaf_values, treedef = jax.tree.flatten(actor)

    return TrainingStateEqxFlatten(
        leaf_values, opt_state, rng, next_env_state, next_obs
    )


@partial(jax.jit, static_argnums=(1, 2, 3))
def collect_experience_eqx_flatten(state, env, env_params, treedef):
    leaf_values, opt_state, rng, env_state, obs = state
    rng, key_act, key_step = jax.random.split(rng, 3)

    actor = jax.tree.unflatten(treedef, leaf_values)

    logits = actor(obs)
    action = jax.random.categorical(key_act, logits)
    next_obs, next_env_state, reward, done, _ = env.step(
        key_step, env_state, action, env_params
    )

    return TrainingStateEqxFlatten(
        leaf_values, opt_state, rng, next_env_state, next_obs
    )


@partial(jax.jit, static_argnums=(1))
def update_policy_eqx_flatten(state, treedef):
    leaf_values, opt_state, rng, env_state, obs = state

    actor = jax.tree.unflatten(treedef, leaf_values)

    grads = jax.tree.map(jnp.zeros_like, eqx.filter(actor, eqx.is_array))
    updates, update_opt_state = optax.adam(1e-3).update(grads, opt_state)
    update_model = eqx.apply_updates(actor, updates)

    update_leaf_values = jax.tree.leaves(update_model)

    return TrainingStateEqxFlatten(
        update_leaf_values, update_opt_state, rng, env_state, obs
    )
