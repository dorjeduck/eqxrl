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
"""Run an actor-critic agent instance on a bsuite experiment."""

from absl import app
from absl import flags

import bsuite

from bsuite.baselines import experiment
from bsuite import sweep
from bsuite.baselines.utils import pool

from typing import Any, Callable, NamedTuple, Tuple

from bsuite.baselines import base
from bsuite.baselines.utils import sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import rlax

import dm_env
from dm_env import specs

"""A simple actor-critic agent implemented in JAX + Equinox."""

Logits = jnp.ndarray
Value = jnp.ndarray
PolicyValueNet = Callable[[jnp.ndarray], Tuple[Logits, Value]]


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


class PolicyValueNetwork(eqx.Module):
    torso: MLP
    policy_head: eqx.nn.Linear
    value_head: eqx.nn.Linear

    def __init__(self, input_size, hidden_sizes, num_actions, key):
        keys = jax.random.split(key, 3)
        layer_sizes = [input_size] + hidden_sizes
        self.torso = MLP(layer_sizes, keys[0])
        self.policy_head = eqx.nn.Linear(hidden_sizes[-1], num_actions, key=keys[1])
        self.value_head = eqx.nn.Linear(hidden_sizes[-1], 1, key=keys[2])

    def __call__(self, inputs):
        flat_inputs = jnp.ravel(inputs)
        embedding = self.torso(flat_inputs)
        logits = self.policy_head(embedding)
        value = self.value_head(embedding)
        return logits, jnp.squeeze(value, axis=-1)
    
    @jax.jit
    def forward_pass(self, *x):
       return jax.vmap(self.__call__)(*x)


class TrainingState(NamedTuple):
    model: PolicyValueNetwork
    opt_state: Any


class ActorCritic(base.Agent):
    """Feed-forward actor-critic agent."""

    def __init__(
        self,
        obs_spec: specs.Array,
        action_spec: specs.DiscreteArray,
        network: PolicyValueNetwork,
        optimizer: optax.GradientTransformation,
        key: jax.Array,
        sequence_length: int,
        discount: float,
        td_lambda: float,
    ):

        # Define loss function.
        def loss(
            model: PolicyValueNetwork, trajectory: sequence.Trajectory
        ) -> jnp.ndarray:
            """ "Actor-critic loss."""
            logits, values = model.forward_pass(trajectory.observations)
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
                w_t=jnp.ones_like(td_errors),
            )

            return actor_loss + critic_loss

        # Define update function.
        @jax.jit
        def sgd_step(
            state: TrainingState, trajectory: sequence.Trajectory
        ) -> TrainingState:
            """Does a step of SGD over a trajectory."""
            gradients = jax.grad(loss)(state.model, trajectory)
            updates, new_opt_state = optimizer.update(gradients, state.opt_state)
            new_model = eqx.apply_updates(state.model, updates)
            return TrainingState(model=new_model, opt_state=new_opt_state)

        # Initialize network and optimiser state.

        initial_opt_state = optimizer.init(eqx.filter(network, eqx.is_array))

        # Internalize state.
        self._state = TrainingState(network, initial_opt_state)
        self._buffer = sequence.Buffer(obs_spec, action_spec, sequence_length)
        self._sgd_step = sgd_step
        self._key = key

    def select_action(self, timestep: dm_env.TimeStep) -> base.Action:
        """Selects actions according to a softmax policy."""
        self._key, action_key = jax.random.split(self._key)
        observation = timestep.observation[None, ...]

        logits, _ = self._state.model.forward_pass(observation)

        action = jax.random.categorical(action_key, logits).squeeze()
        return int(action)

    def update(
        self,
        timestep: dm_env.TimeStep,
        action: base.Action,
        new_timestep: dm_env.TimeStep,
    ):
        """Adds a transition to the trajectory buffer and periodically does SGD."""
        self._buffer.append(timestep, action, new_timestep)
        if self._buffer.full() or new_timestep.last():
            trajectory = self._buffer.drain()
            self._state = self._sgd_step(self._state, trajectory)


def actor_critic_default_agent(
    obs_spec: specs.Array, action_spec: specs.DiscreteArray, seed: int = 0
) -> base.Agent:
    """Creates an actor-critic agent with default hyperparameters."""

    key = jax.random.key(seed)
    input_size = jnp.prod(jnp.array(obs_spec.shape))
    hidden_sizes = [64, 64]

    key, init_key = jax.random.split(key)

    network = PolicyValueNetwork(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        num_actions=action_spec.num_values,
        key=init_key,
    )

    return ActorCritic(
        obs_spec=obs_spec,
        action_spec=action_spec,
        network=network,
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

    agent = actor_critic_default_agent(env.observation_spec(), env.action_spec())

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
