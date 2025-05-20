# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A simple double-DQN agent trained to play BSuite's Catch env."""


import collections
import random
import typing as tp
from absl import app
from absl import flags
from bsuite.environments import catch

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax
import experiment_eqx

Models = collections.namedtuple("Models", "online target")
ActorState = collections.namedtuple("ActorState", "count")
ActorOutput = collections.namedtuple("ActorOutput", "actions q_values")
LearnerState = collections.namedtuple("LearnerState", "count opt_state")
Data = collections.namedtuple("Data", "obs_tm1 a_tm1 r_t discount_t obs_t")

FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("train_episodes", 301, "Number of train episodes.")
flags.DEFINE_integer("batch_size", 32, "Size of the training batch")
flags.DEFINE_float("target_period", 50, "How often to update the target net.")
flags.DEFINE_integer("replay_capacity", 2000, "Capacity of the replay buffer.")
flags.DEFINE_integer("hidden_units", 50, "Number of network hidden units.")
flags.DEFINE_float("epsilon_begin", 1.0, "Initial epsilon-greedy exploration.")
flags.DEFINE_float("epsilon_end", 0.01, "Final epsilon-greedy exploration.")
flags.DEFINE_integer("epsilon_steps", 1000, "Steps over which to anneal eps.")
flags.DEFINE_float("discount_factor", 0.99, "Q-learning discount factor.")
flags.DEFINE_float("learning_rate", 0.005, "Optimizer learning rate.")
flags.DEFINE_integer("eval_episodes", 100, "Number of evaluation episodes.")
flags.DEFINE_integer("evaluate_every", 50, "Number of episodes between evaluations.")


# Define an Equinox MLP for Q-value approximation
class QMLP(eqx.Module):
    layers: tp.List[eqx.nn.Linear]

    def __init__(self, in_size: int, hidden_size: int, out_size: int, key: jax.Array):
        k1, k2 = jax.random.split(key)
        self.layers = [
            eqx.nn.Linear(in_size, hidden_size, key=k1),
            eqx.nn.Linear(hidden_size, out_size, key=k2),
        ]

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.ravel(x)
        x = jax.nn.relu(self.layers[0](x))
        return self.layers[1](x)


# Replace build_network with Equinox version
def build_network(obs_shape, num_actions: int, key):
    in_size = int(np.prod(obs_shape))
    return QMLP(in_size, FLAGS.hidden_units, num_actions, key)


class ReplayBuffer(object):
    """A simple Python replay buffer."""

    def __init__(self, capacity):
        self._prev = None
        self._action = None
        self._latest = None
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, env_output, action):
        self._prev = self._latest
        self._action = action
        self._latest = env_output

        if action is not None:
            self.buffer.append(
                (
                    self._prev.observation,
                    self._action,
                    self._latest.reward,
                    self._latest.discount,
                    self._latest.observation,
                )
            )

    def sample(self, batch_size):
        obs_tm1, a_tm1, r_t, discount_t, obs_t = zip(
            *random.sample(self.buffer, batch_size)
        )
        return (
            np.stack(obs_tm1),
            np.asarray(a_tm1),
            np.asarray(r_t),
            np.asarray(discount_t) * FLAGS.discount_factor,
            np.stack(obs_t),
        )

    def is_ready(self, batch_size):
        return batch_size <= len(self.buffer)


class DQN:
    """A simple DQN agent using Equinox."""

    _target_period: float
    _optimizer: optax.GradientTransformation
    _epsilon_by_frame: tp.Callable[[jnp.ndarray], jnp.ndarray]
    _obs_shape: tuple
    _num_actions: int
    actor_step: tp.Callable
    learner_step: tp.Callable

    def __init__(
        self, observation_spec, action_spec, epsilon_cfg, target_period, learning_rate
    ):
        self._observation_spec = observation_spec
        self._action_spec = action_spec
        self._target_period = target_period
        self._optimizer = optax.adam(learning_rate)
        self._epsilon_by_frame = optax.polynomial_schedule(**epsilon_cfg)
        self._obs_shape = observation_spec.shape
        self._num_actions = action_spec.num_values
        # Jit compile actor_step and learner_step as instance methods
        self.actor_step = jax.jit(self._actor_step)
        self.learner_step = jax.jit(self._learner_step)

    def initial_models(self, key):
    
        online = build_network(self._obs_shape, self._num_actions, key)
        return Models(online, online) #target is same as online at start

    def initial_actor_state(self):
        actor_count = jnp.zeros((), dtype=jnp.float32)
        return ActorState(actor_count)

    def initial_learner_state(self, models):
        learner_count = jnp.zeros((), dtype=jnp.float32)
        opt_state = self._optimizer.init(eqx.filter(models.online, eqx.is_array))
        return LearnerState(learner_count, opt_state)

    def _actor_step(self, models, env_output, actor_state, key, evaluation):
        obs = jnp.expand_dims(env_output.observation, 0)  # add dummy batch
        q = models.online(obs[0])  # Equinox model, single obs
        epsilon = self._epsilon_by_frame(actor_state.count)
        train_a = rlax.epsilon_greedy(epsilon).sample(key, q)
        eval_a = rlax.greedy().sample(key, q)
        a = jax.lax.select(evaluation, eval_a, train_a)
        return ActorOutput(actions=a, q_values=q), ActorState(actor_state.count + 1)

    def _learner_step(self, models, data, learner_state, unused_key):
        target_model = optax.periodic_update(
            models.online, models.target, learner_state.count, self._target_period
        )
        dloss_dtheta = jax.grad(self._loss)(models.online, target_model, *data)
        updates, opt_state = self._optimizer.update(
            dloss_dtheta, learner_state.opt_state
        )
        online_model = eqx.apply_updates(models.online, updates)
        return (
            Models(online_model, target_model),
            LearnerState(learner_state.count + 1, opt_state),
        )

    def _loss(
        self, online_model, target_model, obs_tm1, a_tm1, r_t, discount_t, obs_t
    ):
        # Use vmap for batch processing
        q_tm1 = jax.vmap(online_model)(obs_tm1)
        q_t_val = jax.vmap(target_model)(obs_t)
        q_t_select = jax.vmap(online_model)(obs_t)
        batched_loss = jax.vmap(rlax.double_q_learning)
        td_error = batched_loss(q_tm1, a_tm1, r_t, discount_t, q_t_val, q_t_select)
        return jnp.mean(rlax.l2_loss(td_error))


def main(unused_arg):
    env = catch.Catch(seed=FLAGS.seed)
    epsilon_cfg = dict(
        init_value=FLAGS.epsilon_begin,
        end_value=FLAGS.epsilon_end,
        transition_steps=FLAGS.epsilon_steps,
        power=1.0,
    )
    agent = DQN(
        observation_spec=env.observation_spec(),
        action_spec=env.action_spec(),
        epsilon_cfg=epsilon_cfg,
        target_period=FLAGS.target_period,
        learning_rate=FLAGS.learning_rate,
    )

    accumulator = ReplayBuffer(FLAGS.replay_capacity)
    experiment_eqx.run_loop(
        agent=agent,
        environment=env,
        accumulator=accumulator,
        seed=FLAGS.seed,
        batch_size=FLAGS.batch_size,
        train_episodes=FLAGS.train_episodes,
        evaluate_every=FLAGS.evaluate_every,
        eval_episodes=FLAGS.eval_episodes,
    )


if __name__ == "__main__":
    app.run(main)
