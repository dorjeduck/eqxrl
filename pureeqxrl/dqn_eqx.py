"""
EquinoxRL version of PureJaxRL's DQN: https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/dqn.py
"""

import time
import chex
import equinox as eqx
import flashbax as fbx
import gymnax
import jax
import jax.numpy as jnp
import optax
import wandb
import warnings
import json

from dataclasses import replace
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper

# Enable 64-bit mode in JAX
jax.config.update("jax_enable_x64", True)

# Suppress specific flashbax warning message
warnings.filterwarnings("ignore", message="Setting max_size dynamically*")


class QNetwork(eqx.Module):
    layer1: eqx.nn.Linear
    layer2: eqx.nn.Linear
    layer3: eqx.nn.Linear

    def __init__(self, obs_dim: int, action_dim: int, key):
        keys = jax.random.split(key, 3)
        self.layer1 = eqx.nn.Linear(obs_dim, 120, key=keys[0])
        self.layer2 = eqx.nn.Linear(120, 84, key=keys[1])
        self.layer3 = eqx.nn.Linear(84, action_dim, key=keys[2])

    def __call__(self, x):
        x = jax.nn.relu(self.layer1(x))
        x = jax.nn.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class TrainState(eqx.Module):
    model: eqx.Module
    opt_state: optax.OptState
    tx: optax.GradientTransformation = eqx.static_field()
    step: int

    def apply_gradients(self, grads):
        filtered_model = eqx.filter(self.model, eqx.is_array)
        updates, new_opt_state = self.tx.update(grads, self.opt_state, filtered_model)
        new_model = eqx.apply_updates(self.model, updates)

        return self.replace(
            model=new_model,
            opt_state=new_opt_state,
            step=self.step + 1,
        )

    def replace(self, **kwargs):
        return replace(self, **kwargs)


@chex.dataclass(frozen=True)
class TimeStep:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array


class CustomTrainState(TrainState):
    target_model: eqx.Module
    timesteps: int
    n_updates: int


def make_train(config):

    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_ENVS"]

    basic_env, env_params = gymnax.make(config["ENV_NAME"])
    env = FlattenObservationWrapper(basic_env)
    env = LogWrapper(env)

    vmap_reset = lambda n_envs: lambda rng: jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(rng, n_envs), env_params
    )
    vmap_step = lambda n_envs: lambda rng, env_state, action: jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    )(jax.random.split(rng, n_envs), env_state, action, env_params)

    def train(rng):

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        init_obs, env_state = vmap_reset(config["NUM_ENVS"])(_rng)

        # INIT BUFFER
        buffer = fbx.make_flat_buffer(
            max_length=config["BUFFER_SIZE"],
            min_length=config["BUFFER_BATCH_SIZE"],
            sample_batch_size=config["BUFFER_BATCH_SIZE"],
            add_sequences=False,
            add_batch_size=config["NUM_ENVS"],
        )
        buffer = buffer.replace(
            init=jax.jit(buffer.init),
            add=jax.jit(buffer.add, donate_argnums=0),
            sample=jax.jit(buffer.sample),
            can_sample=jax.jit(buffer.can_sample),
        )
        rng = jax.random.PRNGKey(0)  # use a dummy rng here
        _action = basic_env.action_space().sample(rng)
        _, _env_state = env.reset(rng, env_params)
        _obs, _, _reward, _done, _ = env.step(rng, _env_state, _action, env_params)
        _timestep = TimeStep(obs=_obs, action=_action, reward=_reward, done=_done)
        buffer_state = buffer.init(_timestep)

        # INIT MODEL AND OPTIMIZER
        rng, _rng = jax.random.split(rng)

        model = QNetwork(
            obs_dim=config["OBS_DIM"],  # env.observation_space(env_params).shape[0],
            action_dim=config["ACTION_DIM"],  # env.action_space(env_params).n,
            key=_rng,
        )

        target_model = QNetwork(
            obs_dim=config["OBS_DIM"],
            action_dim=config["ACTION_DIM"],
            key=_rng,
        )

        def linear_schedule(count):
            frac = 1.0 - (count / config["NUM_UPDATES"])
            return config["LR"] * frac

        lr = linear_schedule if config.get("LR_LINEAR_DECAY", False) else config["LR"]
        tx = optax.adam(learning_rate=lr)
        opt_state = tx.init(eqx.filter(model, eqx.is_array))

        train_state = CustomTrainState(
            model=model,
            tx=tx,
            opt_state=opt_state,
            target_model=target_model,
            step=0,
            timesteps=0,
            n_updates=0,
        )

        # epsilon-greedy exploration
        def eps_greedy_exploration(rng, q_vals, t):
            rng_a, rng_e = jax.random.split(
                rng, 2
            )  # a key for sampling random actions and one for picking
            eps = jnp.clip(  # get epsilon
                (
                    (config["EPSILON_FINISH"] - config["EPSILON_START"])
                    / config["EPSILON_ANNEAL_TIME"]
                )
                * t
                + config["EPSILON_START"],
                config["EPSILON_FINISH"],
            )
            greedy_actions = jnp.argmax(q_vals, axis=-1)  # get the greedy actions
            chosed_actions = jnp.where(
                jax.random.uniform(rng_e, greedy_actions.shape)
                < eps,  # pick the actions that should be random
                jax.random.randint(
                    rng_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]
                ),  # sample random actions,
                greedy_actions,
            )
            return chosed_actions

        # TRAINING LOOP
        def _update_step(runner_state, unused):

            train_state, buffer_state, env_state, last_obs, rng = runner_state

            # STEP THE ENV
            rng, rng_a, rng_s = jax.random.split(rng, 3)
            q_vals = jax.vmap(train_state.model)(last_obs)
            action = eps_greedy_exploration(
                rng_a, q_vals, train_state.timesteps
            )  # explore with epsilon greedy_exploration
            obs, env_state, reward, done, info = vmap_step(config["NUM_ENVS"])(
                rng_s, env_state, action
            )
            train_state = train_state.replace(
                timesteps=train_state.timesteps + config["NUM_ENVS"]
            )  # update timesteps count

            # BUFFER UPDATE
            timestep = TimeStep(obs=last_obs, action=action, reward=reward, done=done)
            buffer_state = buffer.add(buffer_state, timestep)

            # NETWORKS UPDATE
            def _learn_phase(train_state, rng):

                learn_batch = buffer.sample(buffer_state, rng).experience

                q_next_target = jax.vmap(train_state.target_model)(
                    learn_batch.second.obs
                )  # (batch_size, num_actions)
                q_next_target = jnp.max(q_next_target, axis=-1)  # (batch_size,)
                target = (
                    learn_batch.first.reward
                    + (1 - learn_batch.first.done) * config["GAMMA"] * q_next_target
                )

                def _loss_fn(model):

                    q_vals = jax.vmap(model)(learn_batch.first.obs)

                    # (batch_size, num_actions)
                    chosen_action_qvals = jnp.take_along_axis(
                        q_vals,
                        jnp.expand_dims(learn_batch.first.action, axis=-1),
                        axis=-1,
                    ).squeeze(axis=-1)
                    return jnp.mean((chosen_action_qvals - target) ** 2)

                loss, grads = jax.value_and_grad(_loss_fn)(train_state.model)
                train_state = train_state.apply_gradients(grads=grads)
                train_state = train_state.replace(n_updates=train_state.n_updates + 1)
                return train_state, loss

            rng, _rng = jax.random.split(rng)
            is_learn_time = (
                (buffer.can_sample(buffer_state))
                & (  # enough experience in buffer
                    train_state.timesteps > config["LEARNING_STARTS"]
                )
                & (  # pure exploration phase ended
                    train_state.timesteps % config["TRAINING_INTERVAL"] == 0
                )  # training interval
            )
            train_state, loss = jax.lax.cond(
                is_learn_time,
                lambda train_state, rng: _learn_phase(train_state, rng),
                lambda train_state, rng: (train_state, jnp.array(0.0)),  # do nothing
                train_state,
                _rng,
            )

            train_state = jax.lax.cond(
                train_state.timesteps % config["TARGET_UPDATE_INTERVAL"] == 0,
                lambda train_state: train_state.replace(
                    target_model=jax.tree.map(
                        lambda source, target: optax.incremental_update(
                            source, target, config["TAU"]
                        ),
                        eqx.filter(train_state.model, eqx.is_array),
                        eqx.filter(train_state.target_model, eqx.is_array),
                    )
                ),
                lambda train_state: train_state,
                operand=train_state,
            )

            metrics = {
                "timesteps": train_state.timesteps,
                "updates": train_state.n_updates,
                "loss": loss.mean(),
                "returns": info["returned_episode_returns"].mean(),
            }

            # report on wandb if required
            if config.get("WANDB_MODE", "disabled") == "online":

                def callback(metrics):
                    if metrics["timesteps"] % 100 == 0:
                        wandb.log(metrics)

                jax.debug.callback(callback, metrics)

            runner_state = (train_state, buffer_state, env_state, obs, rng)

            return runner_state, metrics

        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, buffer_state, env_state, init_obs, _rng)

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train


def main():

    with open("config.json", "r") as f:
        config = json.load(f)

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["DQN", config["ENV_NAME"].upper(), f"jax_{jax.__version__}"],
        name=f'purejaxrl_dqn_{config["ENV_NAME"]}',
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(make_train(config)))

    start = time.time()
    outs = jax.block_until_ready(train_vjit(rngs))
    duration = time.time() - start

    for idx in jnp.lexsort(
        (-jnp.arange(len(outs["metrics"]["returns"][0])), outs["metrics"]["returns"][0])
    )[-10:][::-1]:
        print(f"Index: {idx}, Value: {outs['metrics']['returns'][0][idx]}")

    print(f"Duration: {duration:.2f} seconds")

if __name__ == "__main__":
    main()