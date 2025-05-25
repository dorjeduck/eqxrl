# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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
"""A simple recurrent actor-critic agent implemented in JAX + Haiku."""

from typing import Any, Optional, NamedTuple, Tuple

import bsuite
from bsuite import sweep

from bsuite.baselines import base
from bsuite.baselines.utils import sequence
from bsuite.baselines import experiment

from bsuite.baselines.utils import pool

from absl import app
from absl import flags

import dm_env
from dm_env import specs
import equinox as eqx

import jax
import jax.numpy as jnp
import optax
import rlax

from dataclasses import replace

Logits = jnp.ndarray
Value = jnp.ndarray
LSTMState = Any


class MLP(eqx.Module):
    layers: list

    def __init__(self, layer_sizes, key):
        keys = jax.random.split(key, len(layer_sizes))
        self.layers = []

        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                eqx.nn.Linear(layer_sizes[i], layer_sizes[i + 1], key=keys[i])
            )

    def __call__(self, x):

        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x)


class RecurrentPolicyValueNet(eqx.Module):

    torso: MLP
    lstm: eqx.nn.LSTMCell
    policy_head: eqx.nn.Linear
    value_head: eqx.nn.Linear

    def __init__(
        self,
        in_size: int,
        hidden_size: int,
        num_actions: int,
        key: jax.random.PRNGKey,
    ):
        keys = jax.random.split(key, 4)

        layer_sizes = [in_size] + [hidden_size, hidden_size]
        self.torso = MLP(layer_sizes, keys[0])

        self.lstm = eqx.nn.LSTMCell(hidden_size, hidden_size, key=keys[1])

        self.policy_head = eqx.nn.Linear(
            in_features=hidden_size, out_features=num_actions, key=keys[2]
        )
        self.value_head = eqx.nn.Linear(
            in_features=hidden_size, out_features=1, key=keys[3]
        )

    def __call__(
        self,
        inputs: jnp.ndarray,
        state: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
    ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Processes inputs through the network to obtain policy logits and value estimates.

        Args:
            inputs: jnp.ndarray of shape (seq_len, in_size)
            state: Optional tuple of (hidden_state, cell_state), each of shape (hidden_size,)

        Returns:
            logits: jnp.ndarray of shape (seq_len, num_actions)
            values: jnp.ndarray of shape (seq_len,)
            final_state: Tuple of (hidden_state, cell_state), each of shape (hidden_size,)
        """
        # Pass through MLP torso
        flat_inputs = jnp.ravel(inputs)

        embedding = self.torso(flat_inputs)

        # Process through LSTM
        new_state = self.lstm(embedding, state)

        # Use the LSTM's hidden output, not the input embedding
        lstm_output = new_state[0]  # Extract hidden state

        # Compute policy logits and value estimates using LSTM output
        logits = self.policy_head(lstm_output)
        value = self.value_head(lstm_output)

        return logits, jnp.squeeze(value, axis=-1), new_state

    @jax.jit
    def forward_pass(self, input, state):
        return self.__call__(input, state)


class AgentState(NamedTuple):
    """Holds the network parameters, optimizer state, and RNN state."""

    model: RecurrentPolicyValueNet
    opt_state: Any
    rnn_state: LSTMState
    rnn_unroll_state: LSTMState

    def replace(self, **kwargs):
        return self._replace(**kwargs)


class ActorCriticRNN(base.Agent):
    """Recurrent actor-critic agent."""

    def __init__(
        self,
        obs_spec: specs.Array,
        action_spec: specs.DiscreteArray,
        network: RecurrentPolicyValueNet,
        initial_rnn_state,
        optimizer: optax.GradientTransformation,
        key: jax.random.PRNGKey,
        sequence_length: int,
        discount: float,
        td_lambda: float,
        entropy_cost: float = 0.0,
    ):

        # Define loss function.
        def loss(
            model: RecurrentPolicyValueNet,
            trajectory: sequence.Trajectory,
            rnn_unroll_state: LSTMState,
        ) -> jnp.ndarray:
            """Actor-critic loss."""

            def scan_fn(carry_state, obs):
                logits, value, new_state = model(obs, carry_state)
                return new_state, (logits, value)

            new_rnn_unroll_state, (logits, values) = jax.lax.scan(
                scan_fn, rnn_unroll_state, trajectory.observations
            )
            seq_len = trajectory.actions.shape[0]

            # Remove the ,0 indexing - logits and values are already 1D from scan
            td_errors = rlax.td_lambda(
                v_tm1=values[:-1],
                r_t=trajectory.rewards,
                discount_t=trajectory.discounts * discount,
                v_t=values[1:],
                lambda_=jnp.array(td_lambda),
            )
            critic_loss = jnp.mean(td_errors**2)
            actor_loss = rlax.policy_gradient_loss(
                logits_t=logits[:-1],
                a_t=trajectory.actions,
                adv_t=td_errors,
                w_t=jnp.ones(seq_len),
            )

            entropy_loss = jnp.mean(rlax.entropy_loss(logits[:-1], jnp.ones(seq_len)))

            combined_loss = actor_loss + critic_loss + entropy_cost * entropy_loss

            return combined_loss, new_rnn_unroll_state

        # Define update function.
        @jax.jit
        def sgd_step(state: AgentState, trajectory: sequence.Trajectory) -> AgentState:
            """Does a step of SGD over a trajectory."""
            gradients, new_rnn_state = jax.grad(loss, has_aux=True)(
                state.model, trajectory, state.rnn_unroll_state
            )
            updates, new_opt_state = optimizer.update(gradients, state.opt_state)
            new_model = eqx.apply_updates(state.model, updates)
            return state.replace(
                model=new_model,
                opt_state=new_opt_state,
                rnn_unroll_state=new_rnn_state,
            )

        initial_opt_state = optimizer.init(eqx.filter(network, eqx.is_array))

        # Internalize state.
        self._state = AgentState(
            network, initial_opt_state, initial_rnn_state, initial_rnn_state
        )

        self._buffer = sequence.Buffer(obs_spec, action_spec, sequence_length)
        self._sgd_step = sgd_step
        self._key = key
        self._initial_rnn_state = initial_rnn_state

    def select_action(self, timestep: dm_env.TimeStep) -> base.Action:
        """Selects actions according to a softmax policy."""
        self._key, action_key = jax.random.split(self._key)
        observation = timestep.observation[None, ...]

        logits, _, rnn_state = self._state.model.forward_pass(
            observation, self._state.rnn_state
        )
        self._state = self._state._replace(rnn_state=rnn_state)
        action = jax.random.categorical(action_key, logits).squeeze()
        return int(action)

    def update(
        self,
        timestep: dm_env.TimeStep,
        action: base.Action,
        new_timestep: dm_env.TimeStep,
    ):
        """Adds a transition to the trajectory buffer and periodically does SGD."""
        if new_timestep.last():
            self._state = self._state._replace(rnn_state=self._initial_rnn_state)
        self._buffer.append(timestep, action, new_timestep)
        if self._buffer.full() or new_timestep.last():
            trajectory = self._buffer.drain()
            self._state = self._sgd_step(self._state, trajectory)


def actor_critic_rnn_default_agent(
    obs_spec: specs.Array, action_spec: specs.DiscreteArray, seed: int = 0
) -> base.Agent:
    """Creates an actor-critic agent with default hyperparameters."""

    key = jax.random.key(seed)
    key, model_key = jax.random.split(key)

    hidden_size = 256
    in_size = jnp.prod(jnp.array(obs_spec.shape))

    network = RecurrentPolicyValueNet(
        in_size=in_size,
        hidden_size=hidden_size,
        num_actions=action_spec.num_values,
        key=model_key,
    )

    # Match original: batch dimension of 1
    initial_rnn_state = (
        jnp.zeros((hidden_size), dtype=jnp.float32),
        jnp.zeros((hidden_size), dtype=jnp.float32),
    )

    return ActorCriticRNN(
        obs_spec=obs_spec,
        action_spec=action_spec,
        network=network,
        initial_rnn_state=initial_rnn_state,
        optimizer=optax.adam(3e-3),
        key=key,
        sequence_length=32,
        discount=0.99,
        td_lambda=0.9,
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
flags.DEFINE_integer("num_episodes", None, "Overrides number of training eps.")
flags.DEFINE_boolean("verbose", True, "whether to log to std output")

FLAGS = flags.FLAGS


def run(bsuite_id: str) -> str:
    """Runs an A2C agent on a given bsuite environment, logging to CSV."""

    env = bsuite.load_and_record(
        bsuite_id=bsuite_id,
        save_path=FLAGS.save_path,
        logging_mode=FLAGS.logging_mode,
        overwrite=FLAGS.overwrite,
    )

    agent = actor_critic_rnn_default_agent(env.observation_spec(), env.action_spec())

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
