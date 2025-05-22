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

import dm_env
from dm_env import specs
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import rlax

"""A simple actor-critic agent implemented in JAX + Haiku."""

Logits = jnp.ndarray
Value = jnp.ndarray
PolicyValueNet = Callable[[jnp.ndarray], Tuple[Logits, Value]]


class TrainingState(NamedTuple):
  params: hk.Params
  opt_state: Any


class ActorCritic(base.Agent):
  """Feed-forward actor-critic agent."""

  def __init__(
      self,
      obs_spec: specs.Array,
      action_spec: specs.DiscreteArray,
      network: PolicyValueNet,
      optimizer: optax.GradientTransformation,
      rng: hk.PRNGSequence,
      sequence_length: int,
      discount: float,
      td_lambda: float,
  ):

    # Define loss function.
    def loss(trajectory: sequence.Trajectory) -> jnp.ndarray:
      """"Actor-critic loss."""
      logits, values = network(trajectory.observations)
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
          w_t=jnp.ones_like(td_errors))

      return actor_loss + critic_loss

    # Transform the loss into a pure function.
    loss_fn = hk.without_apply_rng(hk.transform(loss)).apply

    # Define update function.
    @jax.jit
    def sgd_step(state: TrainingState,
                 trajectory: sequence.Trajectory) -> TrainingState:
      """Does a step of SGD over a trajectory."""
      gradients = jax.grad(loss_fn)(state.params, trajectory)
      updates, new_opt_state = optimizer.update(gradients, state.opt_state)
      new_params = optax.apply_updates(state.params, updates)
      return TrainingState(params=new_params, opt_state=new_opt_state)

    # Initialize network parameters and optimiser state.
    init, forward = hk.without_apply_rng(hk.transform(network))
    dummy_observation = jnp.zeros((1, *obs_spec.shape), dtype=jnp.float32)
    initial_params = init(next(rng), dummy_observation)
    initial_opt_state = optimizer.init(initial_params)

    # Internalize state.
    self._state = TrainingState(initial_params, initial_opt_state)
    self._forward = jax.jit(forward)
    self._buffer = sequence.Buffer(obs_spec, action_spec, sequence_length)
    self._sgd_step = sgd_step
    self._rng = rng

  def select_action(self, timestep: dm_env.TimeStep) -> base.Action:
    """Selects actions according to a softmax policy."""
    key = next(self._rng)
    observation = timestep.observation[None, ...]
    
    logits, _ = self._forward(self._state.params, observation)
    action = jax.random.categorical(key, logits).squeeze()
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


def actor_critic_default_agent(obs_spec: specs.Array,
                  action_spec: specs.DiscreteArray,
                  seed: int = 0) -> base.Agent:
  """Creates an actor-critic agent with default hyperparameters."""

  def network(inputs: jnp.ndarray) -> Tuple[Logits, Value]:
    flat_inputs = hk.Flatten()(inputs)
    torso = hk.nets.MLP([64, 64])
    policy_head = hk.Linear(action_spec.num_values)
    value_head = hk.Linear(1)
    embedding = torso(flat_inputs)
    logits = policy_head(embedding)
    value = value_head(embedding)
    return logits, jnp.squeeze(value, axis=-1)

  return ActorCritic(
      obs_spec=obs_spec,
      action_spec=action_spec,
      network=network,
      optimizer=optax.adam(3e-3),
      rng=hk.PRNGSequence(seed),
      sequence_length=32,
      discount=0.99,
      td_lambda=0.9,
  )


# Experiment flags.
flags.DEFINE_string(
    'bsuite_id', 'catch/0', 'BSuite identifier. '
    'This global flag can be used to control which environment is loaded.')
flags.DEFINE_string('save_path', '/tmp/bsuite', 'where to save bsuite results')
flags.DEFINE_enum('logging_mode', 'csv', ['csv', 'sqlite', 'terminal'],
                  'which form of logging to use for bsuite results')
flags.DEFINE_boolean('overwrite', True, 'overwrite csv logging if found')
flags.DEFINE_integer('num_episodes', None, 'Overrides number of training eps.')
flags.DEFINE_boolean('verbose', True, 'whether to log to std output')

FLAGS = flags.FLAGS


def run(bsuite_id: str) -> str:
  """Runs an A2C agent on a given bsuite environment, logging to CSV."""

  env = bsuite.load_and_record(
      bsuite_id=bsuite_id,
      save_path=FLAGS.save_path,
      logging_mode=FLAGS.logging_mode,
      overwrite=FLAGS.overwrite,
  )

  agent = actor_critic_default_agent(
      env.observation_spec(), env.action_spec())

  num_episodes = FLAGS.num_episodes or getattr(env, 'bsuite_num_episodes')
  experiment.run(
      agent=agent,
      environment=env,
      num_episodes=num_episodes,
      verbose=FLAGS.verbose)

  return bsuite_id


def main(_):
  # Parses whether to run a single bsuite_id, or multiprocess sweep.
  bsuite_id = FLAGS.bsuite_id

  if bsuite_id in sweep.SWEEP:
    print(f'Running single experiment: bsuite_id={bsuite_id}.')
    run(bsuite_id)

  elif hasattr(sweep, bsuite_id):
    bsuite_sweep = getattr(sweep, bsuite_id)
    print(f'Running sweep over bsuite_id in sweep.{bsuite_sweep}')
    FLAGS.verbose = False
    pool.map_mpi(run, bsuite_sweep)

  else:
    raise ValueError(f'Invalid flag: bsuite_id={bsuite_id}.')


if __name__ == '__main__':
  app.run(main)