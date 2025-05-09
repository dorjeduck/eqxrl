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


class TrainingStateEqx(NamedTuple):
    actor: ActorEqx
    opt_state: optax.OptState
    rng: jax.random.PRNGKey
    env_state: any
    obs: jnp.ndarray


@jax.jit
def forward_pass_eqx(state: TrainingStateEqx, _):
    return state.actor(state.obs)


@partial(jax.jit, static_argnums=(1, 2))
def collect_experience_eqx(state, env, env_params, _):
    actor, opt_state, rng, env_state, obs = state
    rng, key_act, key_step = jax.random.split(rng, 3)

    logits = actor(obs)

    action = jax.random.categorical(key_act, logits)
    next_obs, next_env_state, reward, done, _ = env.step(
        key_step, env_state, action, env_params
    )

    return TrainingStateEqx(actor, opt_state, rng, next_env_state, next_obs)


@partial(jax.jit, static_argnums=(1, 2))
def collect_experience_eqx(state, env, env_params, _):
    actor, opt_state, rng, env_state, obs = state
    rng, key_act, key_step = jax.random.split(rng, 3)

    logits = actor(obs)
    action = jax.random.categorical(key_act, logits)
    next_obs, next_env_state, reward, done, _ = env.step(
        key_step, env_state, action, env_params
    )

    return TrainingStateEqx(actor, opt_state, rng, next_env_state, next_obs)


@jax.jit
def update_policy_eqx(state):
    actor, opt_state, rng, env_state, obs = state

    grads = jax.tree.map(jnp.zeros_like, eqx.filter(actor, eqx.is_array))
    updates, new_opt_state = optax.adam(1e-3).update(grads, opt_state)
    new_actor = eqx.apply_updates(actor, updates)

    return TrainingStateEqx(new_actor, new_opt_state, rng, env_state, obs)
