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
"""A simple JAX-based DQN implementation.

Reference: "Playing atari with deep reinforcement learning" (Mnih et al, 2015).
Link: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf.
"""

from typing import Any, Callable, NamedTuple, Sequence

import bsuite
from bsuite import sweep

from bsuite.baselines import experiment
from bsuite.baselines.jax import dqn
from bsuite.baselines.utils import pool

from bsuite.baselines import base
from bsuite.baselines.utils import replay

import dm_env
from dm_env import specs
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax
import typing as tp

from absl import app
from absl import flags


class QMLP(eqx.Module):
    layers: tp.List[eqx.nn.Linear]

    def __init__(self, in_size: int, hidden_size: int, out_size: int, key: jax.Array):
        k1, k2, k3 = jax.random.split(key, 3)
        self.layers = [
            eqx.nn.Linear(in_size, hidden_size, key=k1),
            eqx.nn.Linear(in_size, hidden_size, key=k2),
            eqx.nn.Linear(hidden_size, out_size, key=k3),
        ]

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.ravel(x)
        x = jax.nn.relu(self.layers[0](x))
        x = jax.nn.relu(self.layers[1](x))
        return self.layers[2](x)

    @jax.jit
    def forward_pass(self, *x):
        return jax.vmap(self.__call__)(*x)


class TrainingState(NamedTuple):
    """Holds the agent's training state."""

    model: QMLP
    target_params: QMLP
    opt_state: Any
    step: int


class DQN(base.Agent):
    """A simple DQN agent using JAX."""

    def __init__(
        self,
        obs_spec: specs.Array,
        action_spec: specs.DiscreteArray,
        model: QMLP,
        target_model: QMLP,
        optimizer: optax.GradientTransformation,
        batch_size: int,
        epsilon: float,
        discount: float,
        replay_capacity: int,
        min_replay_size: int,
        sgd_period: int,
        target_update_period: int,
    ):

        # Define loss function.
        def loss(
            model: QMLP, target_model: QMLP, transitions: Sequence[jnp.ndarray]
        ) -> jnp.ndarray:
            """Computes the standard TD(0) Q-learning loss on batch of transitions."""
            o_tm1, a_tm1, r_t, d_t, o_t = transitions
            q_tm1 = jax.vmap(model)(o_tm1)
            q_t = jax.vmap(target_model)(o_t)

            batch_q_learning = jax.vmap(rlax.q_learning)
            td_error = batch_q_learning(q_tm1, a_tm1, r_t, discount * d_t, q_t)
            return jnp.mean(td_error**2)

        # Define update function.
        @jax.jit
        def sgd_step(
            state: TrainingState, transitions: Sequence[jnp.ndarray]
        ) -> TrainingState:
            """Performs an SGD step on a batch of transitions."""
            gradients = jax.grad(loss)(state.model, state.target_model, transitions)
            updates, new_opt_state = optimizer.update(gradients, state.opt_state)
            new_model = eqx.apply_updates(state.model, updates)

            return TrainingState(
                model=new_model,
                target_model=state.target_model,
                opt_state=new_opt_state,
                step=state.step + 1,
            )

        # Initialize optimiser state.

        initial_opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

        # This carries the agent state relevant to training.
        self._state = TrainingState(
            model=model,
            target_model=target_model,  # Use the same model for the target model initial
            opt_state=initial_opt_state,
            step=0,
        )
        self._sgd_step = sgd_step

        self._replay = replay.Replay(capacity=replay_capacity)

        # Store hyperparameters.
        self._num_actions = action_spec.num_values
        self._batch_size = batch_size
        self._sgd_period = sgd_period
        self._target_update_period = target_update_period
        self._epsilon = epsilon
        self._total_steps = 0
        self._min_replay_size = min_replay_size

    def select_action(self, timestep: dm_env.TimeStep) -> base.Action:
        """Selects actions according to an epsilon-greedy policy."""
        if np.random.rand() < self._epsilon:
            return np.random.randint(self._num_actions)

        # Greedy policy, breaking ties uniformly at random.
        observation = timestep.observation[None, ...]
        q_values = self.state.model.forward_pass(observation)

        action = np.random.choice(np.flatnonzero(q_values == q_values.max()))
        return int(action)

    def update(
        self,
        timestep: dm_env.TimeStep,
        action: base.Action,
        new_timestep: dm_env.TimeStep,
    ):
        """Adds transition to replay and periodically does SGD."""
        # Add this transition to replay.
        self._replay.add(
            [
                timestep.observation,
                action,
                new_timestep.reward,
                new_timestep.discount,
                new_timestep.observation,
            ]
        )

        self._total_steps += 1
        if self._total_steps % self._sgd_period != 0:
            return

        if self._replay.size < self._min_replay_size:
            return

        # Do a batch of SGD.
        transitions = self._replay.sample(self._batch_size)
        self._state = self._sgd_step(self._state, transitions)

        # Periodically update target parameters.
        if self._state.step % self._target_update_period == 0:
            self._state = self._state._replace(target_params=self._state.params)


def default_agent(
    obs_spec: specs.Array, action_spec: specs.DiscreteArray, seed: int = 0
) -> base.Agent:
    """Initialize a DQN agent with default parameters."""

    model_key, target_key = jax.random.key(seed)

    model = QMLP(
        in_size=np.prod(obs_spec.shape),
        hidden_size=64,
        out_size=action_spec.num_values,
        key=model_key,
    )

    # using a differently initialized target model as in the original BSuite DQN implementation
    target_model = QMLP(
        in_size=np.prod(obs_spec.shape),
        hidden_size=64,
        out_size=action_spec.num_values,
        key=target_key,
    )

    return DQN(
        obs_spec=obs_spec,
        action_spec=action_spec,
        model=model,
        target_model=target_model,
        optimizer=optax.adam(1e-3),
        batch_size=32,
        discount=0.99,
        replay_capacity=10000,
        min_replay_size=100,
        sgd_period=1,
        target_update_period=4,
        epsilon=0.05,
    )


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

    agent = dqn.default_agent(env.observation_spec(), env.action_spec())

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
