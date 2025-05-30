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
"""Experiment loop."""


import jax


def run_loop(
    agent, environment, accumulator, seed,
    batch_size, train_episodes, evaluate_every, eval_episodes):
  """A simple run loop for examples of reinforcement learning with rlax."""

  # Init agent.
  key = jax.random.key(seed)

  key,models_key = jax.random.split(key)

  models = agent.initial_models(models_key)
  learner_state = agent.initial_learner_state(models)

  print(f"Training agent for {train_episodes} episodes")
  for episode in range(train_episodes):

    # Prepare agent, environment and accumulator for a new episode.
    timestep = environment.reset()
    accumulator.push(timestep, None)
    actor_state = agent.initial_actor_state()

    while not timestep.last():

      # Acting.
      key,next_key = jax.random.split(key)
      actor_output, actor_state = agent.actor_step(
          models, timestep, actor_state, next_key, evaluation=False)

      # Agent-environment interaction.
      action = int(actor_output.actions)
      timestep = environment.step(action)

      # Accumulate experience.
      accumulator.push(timestep, action)

      # Learning.
      if accumulator.is_ready(batch_size):
        key,next_key = jax.random.split(key)
        models, learner_state = agent.learner_step(
            models, accumulator.sample(batch_size), learner_state, next_key)

    # Evaluation.
    if not episode % evaluate_every:
      returns = 0.
      for _ in range(eval_episodes):
        timestep = environment.reset()
        actor_state = agent.initial_actor_state()

        while not timestep.last():
          key,next_key = jax.random.split(key)
          actor_output, actor_state = agent.actor_step(
              models, timestep, actor_state, next_key, evaluation=True)
          timestep = environment.step(int(actor_output.actions))
          returns += timestep.reward

      avg_returns = returns / eval_episodes
      print(f"Episode {episode:4d}: Average returns: {avg_returns:.2f}")
