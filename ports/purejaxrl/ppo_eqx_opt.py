import jax
import jax.numpy as jnp

import optax
import json
import time
from flax.linen.initializers import constant, orthogonal
from typing import NamedTuple, Any
from jaxtyping import PyTreeDef

from flax.training.train_state import TrainState
import distrax
import gymnax
from wrappers import LogWrapper, FlattenObservationWrapper

from dataclasses import replace
import equinox as eqx


class ActorCritic(eqx.Module):

    activation_fn: Any = eqx.static_field()
    actor_mean_layer1: eqx.nn.Linear
    actor_mean_layer2: eqx.nn.Linear
    actor_mean_layer3: eqx.nn.Linear
    critic_layer1: eqx.nn.Linear
    critic_layer2: eqx.nn.Linear
    critic_layer3: eqx.nn.Linear

    def __init__(self, obs_dim: int, action_dim: int, activation, key):

        self.activation_fn = jax.nn.relu if activation == "relu" else jax.nn.tanh

        keys = jax.random.split(key, 6)

        self.actor_mean_layer1 = eqx.nn.Linear(obs_dim, 64, key=keys[0])
        self.actor_mean_layer2 = eqx.nn.Linear(64, 64, key=keys[1])
        self.actor_mean_layer3 = eqx.nn.Linear(64, action_dim, key=keys[2])
        self.critic_layer1 = eqx.nn.Linear(obs_dim, 64, key=keys[3])
        self.critic_layer2 = eqx.nn.Linear(64, 64, key=keys[4])
        self.critic_layer3 = eqx.nn.Linear(64, 1, key=keys[5])

    def __call__(self, x):

        actor_mean = self.activation_fn(self.actor_mean_layer1(x))
        actor_mean = self.activation_fn(self.actor_mean_layer2(actor_mean))
        actor_mean = self.actor_mean_layer3(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = self.activation_fn(self.critic_layer1(x))
        critic = self.activation_fn(self.critic_layer2(critic))
        critic = self.critic_layer3(critic)

        return pi, jnp.squeeze(critic, axis=-1)


class TrainState(eqx.Module):
    flat_model: list
    flat_opt_state: list

    treedef_model: PyTreeDef = eqx.static_field()
    treedef_opt_state: PyTreeDef = eqx.static_field()

    tx: optax.GradientTransformation = eqx.static_field()

    step: int

    def apply_gradients(self, grads):

        model = jax.tree.unflatten(self.treedef_model, self.flat_model)
        opt_state = jax.tree.unflatten(self.treedef_opt_state, self.flat_opt_state)

        updates, update_opt_state = self.tx.update(grads, opt_state)
        update_model = eqx.apply_updates(model, updates)

        flat_update_model = jax.tree.leaves(update_model)
        flat_update_opt_state = jax.tree.leaves(update_opt_state)

        return self.replace(
            flat_model=flat_update_model,
            flat_opt_state=flat_update_opt_state,
            step=self.step + 1,
        )

    def replace(self, **kwargs):
        return replace(self, **kwargs)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    env, env_params = gymnax.make(config["ENV_NAME"])
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK

        rng, _rng = jax.random.split(rng)

        model = ActorCritic(
            obs_dim=config["OBS_DIM"],
            action_dim=config["ACTION_DIM"],
            activation=config["ACTIVATION"],
            key=_rng,
        )

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )

        opt_state = tx.init(eqx.filter(model, eqx.is_array))

        flat_model, treedef_model = jax.tree.flatten(model)
        flat_opt_state, treedef_opt_state = jax.tree.flatten(opt_state)

        train_state = TrainState(
            flat_model=flat_model,
            flat_opt_state=flat_opt_state,
            treedef_model=treedef_model,
            treedef_opt_state=treedef_opt_state,
            tx=tx,
            step=0,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                model = jax.tree.unflatten(
                    train_state.treedef_model, train_state.flat_model
                )
                pi, value = jax.vmap(model)(last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state

            model = jax.tree.unflatten(
                train_state.treedef_model, train_state.flat_model
            )

            _, last_val = jax.vmap(model)(last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(model, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = jax.vmap(model)(traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)

                    model = jax.tree.unflatten(
                        train_state.treedef_model, train_state.flat_model
                    )

                    total_loss, grads = grad_fn(model, traj_batch, advantages, targets)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                # Batching and Shuffling
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                # Mini-batch Updates
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            # Updating Training State and Metrics:
            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            # Debugging mode
            if config.get("DEBUG"):

                def callback(info):
                    return_values = info["returned_episode_returns"][
                        info["returned_episode"]
                    ]
                    timesteps = (
                        info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    )
                    for t in range(len(timesteps)):
                        print(
                            f"global step={timesteps[t]}, episodic return={return_values[t]}"
                        )

                jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


if __name__ == "__main__":

    with open("./config/ppo_config.json", "r") as f:
        config = json.load(f)

    rng = jax.random.key(30)
    train_jit = jax.jit(make_train(config))

    if config["BENCHMARK"]:
        # warmup
        if config["BENCHMARK_WARMUP"]:
            _ = jax.block_until_ready(train_jit(rng))

        start = time.time()
        durations = []
        for _ in range(config["BENCHMARK_ROUNDS"]):
            out = jax.block_until_ready(train_jit(rng))
            durations.append(time.time() - start)
            start = time.time()
        average_duration = sum(durations) / len(durations)
        print(f"Average Duration: {average_duration:.2f} seconds")
    else:
        _ = jax.block_until_ready(train_jit(rng))
