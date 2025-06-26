#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""A simple implementation of Bootstrapped DQN with prior networks.

References:
1. "Deep Exploration via Bootstrapped DQN" (Osband et al., 2016)
2. "Deep Exploration via Randomized Value Functions" (Osband et al., 2017)
3. "Randomized Prior Functions for Deep RL" (Osband et al, 2018)

Links:
1. https://arxiv.org/abs/1602.04621
2. https://arxiv.org/abs/1703.07608
3. https://arxiv.org/abs/1806.03335

Notes:

- This agent is implemented with TensorFlow 2 and Sonnet 2. For installation
  instructions for these libraries, see the README.md in the parent folder.
- This implementation is potentially inefficient, as it does not parallelise
  computation across the ensemble for simplicity and readability.
"""

from typing import Any, Callable, NamedTuple, Sequence

import bsuite
from bsuite import sweep

from bsuite.baselines import experiment
from bsuite.baselines.jax import boot_dqn
from bsuite.baselines.utils import pool
from bsuite.baselines import base
from bsuite.baselines.utils import replay

import dm_env
from dm_env import specs
import equinox as eqx
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import optax
import rlax
import typing as tp

from absl import app
from absl import flags

import haiku as hk
from jax import lax
import jax.numpy as jnp
import optax


class QMLPWithPrior(eqx.Module):
    layers: tp.List[eqx.nn.Linear]
    prior_layers: tp.List[eqx.nn.Linear]
    prior_scale: float = eqx.static_field()

    def __init__(
        self,
        in_size: int,
        hidden_size: int,
        out_size: int,
        key: jax.Array,
        prior_scale: float = 5.0,
    ):
        # Split keys for main network and prior network
        main_key, prior_key = jax.random.split(key)
        k1, k2, k3 = jax.random.split(main_key, 3)
        pk1, pk2, pk3 = jax.random.split(prior_key, 3)

        # Main trainable network
        self.layers = [
            eqx.nn.Linear(in_size, hidden_size, key=k1),
            eqx.nn.Linear(hidden_size, hidden_size, key=k2),
            eqx.nn.Linear(hidden_size, out_size, key=k3),
        ]

        # Prior network (will be frozen)
        self.prior_layers = [
            eqx.nn.Linear(in_size, hidden_size, key=pk1),
            eqx.nn.Linear(hidden_size, hidden_size, key=pk2),
            eqx.nn.Linear(hidden_size, out_size, key=pk3),
        ]

        self.prior_scale = prior_scale

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.ravel(x)

        # Main network forward pass
        main_x = jax.nn.relu(self.layers[0](x))
        main_x = jax.nn.relu(self.layers[1](main_x))
        main_output = self.layers[2](main_x)

        # Prior network forward pass (with stop_gradient)
        prior_x = jax.nn.relu(self.prior_layers[0](x))
        prior_x = jax.nn.relu(self.prior_layers[1](prior_x))
        prior_output = self.prior_layers[2](prior_x)

        # Combine main output with scaled prior
        return main_output + self.prior_scale * prior_output

    @jax.jit
    def forward_pass(self, *x):
        return jax.vmap(self.__call__)(*x)


class TrainingState(NamedTuple):
    model: QMLPWithPrior
    target_model: QMLPWithPrior
    opt_state: Any
    step: int


class BootstrappedDqn(base.Agent):
    """Bootstrapped DQN with randomized prior functions."""

    def __init__(
        self,
        action_spec: specs.DiscreteArray,
        models: tp.List[QMLPWithPrior],
        target_models: tp.List[QMLPWithPrior],
        num_ensemble: int,
        batch_size: int,
        discount: float,
        replay_capacity: int,
        min_replay_size: int,
        sgd_period: int,
        target_update_period: int,
        optimizer: optax.GradientTransformation,
        mask_prob: float,
        noise_scale: float,
        epsilon_fn: Callable[[int], float] = lambda _: 0.0,
    ):

        # Define loss function, including bootstrap mask `m_t` & reward noise `z_t`.
        def loss(
            diff_model,
            static_model,
            target_model: QMLPWithPrior,
            transitions: Sequence[jnp.ndarray],
        ) -> jnp.ndarray:
            """Q-learning loss with added reward noise + half-in bootstrap."""

            model = eqx.combine(diff_model, static_model)
            o_tm1, a_tm1, r_t, d_t, o_t, m_t, z_t = transitions
            q_tm1 = jax.vmap(model)(o_tm1)
            q_t = jax.vmap(target_model)(o_t)
            r_t += noise_scale * z_t
            batch_q_learning = jax.vmap(rlax.q_learning)
            td_error = batch_q_learning(q_tm1, a_tm1, r_t, discount * d_t, q_t)
            return jnp.mean(m_t * td_error**2)

        # Define update function for each member of ensemble..
        @jax.jit
        def sgd_step(
            state: TrainingState, transitions: Sequence[jnp.ndarray]
        ) -> TrainingState:
            """Does a step of SGD for the whole ensemble over `transitions`."""

            diff_model, static_model = eqx.partition(state.model, self._filter_spec)

            gradients = eqx.filter_grad(loss)(
                diff_model, static_model, state.target_model, transitions
            )
            updates, new_opt_state = optimizer.update(gradients, state.opt_state)
            new_model = eqx.apply_updates(state.model, updates)

            return TrainingState(
                model=new_model,
                target_model=state.target_model,
                opt_state=new_opt_state,
                step=state.step + 1,
            )

        # filter for frozen weigths of the prior_layers

        self._filter_spec = jax.tree_util.tree.map(lambda _: True, models[0])
        self._filter_spec = eqx.tree_at(
            lambda tree: [
                w for layer in tree.prior_layers for w in (layer.weight, layer.bias)
            ],
            self._filter_spec,
            replace=[False] * (2 * len(models[0].prior_layers)),
        )

        # Initialize parameters and optimizer state for an ensemble of Q-networks.

        initial_opt_state = [
            optimizer.init(eqx.filter(model, eqx.is_array)) for model in models
        ]

        # Internalize state.
        self._ensemble = [
            TrainingState(p, tp, o, step=0)
            for p, tp, o in zip(models, target_models, initial_opt_state)
        ]

        self._sgd_step = sgd_step
        self._num_ensemble = num_ensemble
        self._optimizer = optimizer
        self._replay = replay.Replay(capacity=replay_capacity)

        # Agent hyperparameters.
        self._num_actions = action_spec.num_values
        self._batch_size = batch_size
        self._sgd_period = sgd_period
        self._target_update_period = target_update_period
        self._min_replay_size = min_replay_size
        self._epsilon_fn = epsilon_fn
        self._mask_prob = mask_prob

        # Agent state.
        self._active_head = self._ensemble[0]
        self._total_steps = 0

    def select_action(self, timestep: dm_env.TimeStep) -> base.Action:
        """Select values via Thompson sampling, then use epsilon-greedy policy."""
        self._total_steps += 1
        if np.random.rand() < self._epsilon_fn(self._total_steps):
            return np.random.randint(self._num_actions)

        # Greedy policy, breaking ties uniformly at random.
        batched_obs = timestep.observation[None, ...]
        q_values = self._active_head.model.forward_pass(batched_obs)

        action = np.random.choice(np.flatnonzero(q_values == q_values.max()))
        return int(action)

    def update(
        self,
        timestep: dm_env.TimeStep,
        action: base.Action,
        new_timestep: dm_env.TimeStep,
    ):
        """Update the agent: add transition to replay and periodically do SGD."""

        # Thompson sampling: every episode pick a new Q-network as the policy.
        if new_timestep.last():
            k = np.random.randint(self._num_ensemble)
            self._active_head = self._ensemble[k]

        # Generate bootstrapping mask & reward noise.
        mask = np.random.binomial(1, self._mask_prob, self._num_ensemble)
        noise = np.random.randn(self._num_ensemble)

        # Make transition and add to replay.
        transition = [
            timestep.observation,
            action,
            np.float32(new_timestep.reward),
            np.float32(new_timestep.discount),
            new_timestep.observation,
            mask,
            noise,
        ]
        self._replay.add(transition)

        if self._replay.size < self._min_replay_size:
            return

        # Periodically sample from replay and do SGD for the whole ensemble.
        if self._total_steps % self._sgd_period == 0:
            transitions = self._replay.sample(self._batch_size)
            o_tm1, a_tm1, r_t, d_t, o_t, m_t, z_t = transitions
            for k, state in enumerate(self._ensemble):
                transitions = [o_tm1, a_tm1, r_t, d_t, o_t, m_t[:, k], z_t[:, k]]
                self._ensemble[k] = self._sgd_step(state, transitions)

        # Periodically update target parameters.
        for k, state in enumerate(self._ensemble):
            if state.step % self._target_update_period == 0:
                self._ensemble[k] = state._replace(target_model=state.model)


def default_agent(
    obs_spec: specs.Array,
    action_spec: specs.DiscreteArray,
    seed: int = 0,
    num_ensemble: int = 20,
) -> BootstrappedDqn:
    """Initialize a Bootstrapped DQN agent with default parameters."""

    # Define network.
    prior_scale = 5.0
    hidden_sizes = [50, 50]

    keys = jax.random.key(seed, 2 * num_ensemble)

    models = [
        QMLPWithPrior(
            in_size=np.prod(obs_spec.shape),
            hidden_size=64,
            out_size=action_spec.num_values,
            key=keys[i],
        )
        for i in range(num_ensemble)
    ]

    target_models = [
        QMLPWithPrior(
            in_size=np.prod(obs_spec.shape),
            hidden_size=64,
            out_size=action_spec.num_values,
            key=keys[i + num_ensemble],
        )
        for i in range(num_ensemble)
    ]

    optimizer = optax.adam(learning_rate=1e-3)
    return BootstrappedDqn(
        action_spec=action_spec,
        models=models,
        target_models=target_models,
        batch_size=128,
        discount=0.99,
        num_ensemble=num_ensemble,
        replay_capacity=10000,
        min_replay_size=128,
        sgd_period=1,
        target_update_period=4,
        optimizer=optimizer,
        mask_prob=1.0,
        noise_scale=0.0,
        epsilon_fn=lambda _: 0.0,
    )


# Internal imports.

flags.DEFINE_integer("num_ensemble", 1, "Size of ensemble.")

# Experiment flags.
flags.DEFINE_string(
    "bsuite_id",
    "catch/0",
    "BSuite identifier. "
    "This global flag can be used to control which environment is loaded.",
)
flags.DEFINE_string("save_path", "/tmp/bsuite", "where to save bsuite results")
flags.DEFINE_enum(
    "logging_mode",
    "csv",
    ["csv", "sqlite", "terminal"],
    "which form of logging to use for bsuite results",
)
flags.DEFINE_boolean("overwrite", True, "overwrite csv logging if found")
flags.DEFINE_integer("num_episodes", None, "Number of episodes to run for.")
flags.DEFINE_boolean("verbose", True, "whether to log to std output")

FLAGS = flags.FLAGS


def run(bsuite_id: str) -> str:
    """Runs a DQN agent on a given bsuite environment, logging to CSV."""

    env = bsuite.load_and_record(
        bsuite_id=bsuite_id,
        save_path=FLAGS.save_path,
        logging_mode=FLAGS.logging_mode,
        overwrite=FLAGS.overwrite,
    )
    obs_spec = env.observation_spec()
    action_spec = env.action_spec()

    # Define network.
    prior_scale = 5.0
    hidden_size = 50
    key = jax.random.key(1)
    num_ensemble = FLAGS.num_ensemble

    keys = jax.random.split(key, 2 * num_ensemble)

    models = [
        QMLPWithPrior(
            in_size=np.prod(obs_spec.shape),
            hidden_size=hidden_size,
            out_size=action_spec.num_values,
            key=keys[i],
            prior_scale=prior_scale,
        )
        for i in range(num_ensemble)
    ]

    target_models = [
        QMLPWithPrior(
            in_size=np.prod(obs_spec.shape),
            hidden_size=hidden_size,
            out_size=action_spec.num_values,
            key=keys[i + num_ensemble],
            prior_scale=prior_scale,
        )
        for i in range(num_ensemble)
    ]

    optimizer = optax.adam(learning_rate=1e-3)

    agent = BootstrappedDqn(
        action_spec=action_spec,
        models=models,
        target_models=target_models,
        optimizer=optimizer,
        num_ensemble=FLAGS.num_ensemble,
        batch_size=128,
        discount=0.99,
        replay_capacity=10000,
        min_replay_size=128,
        sgd_period=1,
        target_update_period=4,
        mask_prob=1.0,
        noise_scale=0.0,
    )

    num_episodes = FLAGS.num_episodes or getattr(env, "bsuite_num_episodes")
    experiment.run(
        agent=agent, environment=env, num_episodes=num_episodes, verbose=FLAGS.verbose
    )

    return bsuite_id


def main(_):
    # Parses whether to run a single bsuite_id, or multiprocess sweep.
    bsuite_id = FLAGS.bsuite_id

    if bsuite_id in sweep.SWEEP:
        print(f"Running single experiment: bsuite_id={bsuite_id}.")
        run(bsuite_id)

    elif hasattr(sweep, bsuite_id):
        bsuite_sweep = getattr(sweep, bsuite_id)
        print(f"Running sweep over bsuite_id in sweep.{bsuite_sweep}")
        FLAGS.verbose = False
        pool.map_mpi(run, bsuite_sweep)

    else:
        raise ValueError(f"Invalid flag: bsuite_id={bsuite_id}.")


if __name__ == "__main__":
    app.run(main)
