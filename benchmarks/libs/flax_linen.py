from typing import NamedTuple
import jax
import jax.numpy as jnp
import optax

from flax import linen as nn
from functools import partial


class ActorLinen(nn.Module):
    action_size: int

    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(64)(x))
        x = nn.relu(nn.Dense(64)(x))
        x = nn.Dense(self.action_size)(x)
        return x


class TrainingStateLinen(NamedTuple):
    params: nn.Module
    opt_state: optax.OptState
    rng: jax.random.PRNGKey
    env_state: any
    obs: jnp.ndarray


@partial(jax.jit, static_argnums=(1))
def forward_pass_linen(state, actor):
    return actor.apply(state.params, state.obs)


@partial(jax.jit, static_argnums=(1, 2, 3))
def collect_experience_linen(state, env, env_params, actor):
    params, opt_state, rng, env_state, obs = state
    rng, key_act, key_step = jax.random.split(rng, 3)

    logits = actor.apply(params, obs)
    action = jax.random.categorical(key_act, logits)
    next_obs, next_env_state, reward, done, _ = env.step(
        key_step, env_state, action, env_params
    )

    return TrainingStateLinen(params, opt_state, rng, next_env_state, next_obs)


@partial(jax.jit, static_argnums=(1))
def update_policy_linen(state, _):
    params, opt_state, rng, env_state, obs = state

    grads = jax.tree_util.tree.map(jnp.zeros_like, params)
    updates, new_opt_state = optax.adam(1e-3).update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)

    return TrainingStateLinen(new_params, new_opt_state, rng, env_state, obs)
