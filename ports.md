
# Ports

Ports of Deep Reinforcement Learning implementations to Jax/Equinox.

* [CleanRL](https://github.com/vwxyzjn/cleanrl)

  * [**c51_jax_eqx**](./ports/cleanrl/c51_jax_eqx): Port of the CleanRL Categorical DQN implementation [c51_jax.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/c51_jax.py) from Flax/Linen to Equinox.
  
  * [**ddpg_continuous_action_jax_eqx**](./ports/cleanrl/ddpg_continuous_action_jax_eqx.py): Port of the CleanRL DDPG implementation [ddpg_continuous_action_jax.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ddpg_continuous_action_jax.py) from Flax/Linen to Equinox.
  
  * [**dqn_jax_eqx.py**](./ports/cleanrl/dqn_jax_eqx.py): Port of the CleanRL DQN implementation [dqn_jax.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_jax.py) from Flax/Linen to Equinox.
  
  * [**dqn_atari_jax_eqx.py**](./ports/cleanrl/dqn_atari_jax_eqx.py): Port of the CleanRL DQN implementation [dqn_atari_jax.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari_jax.py) from Flax/Linen to Equinox.

  * [**td3_continuous_action_jax_eqx**](./ports/cleanrl/td3_continuous_action_jax_eqx.py): Port of the CleanRL Twin Delayed Deep Deterministic Policy Gradient implementation [td3_continuous_action_jax.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td3_continuous_action_jax.py) from Flax/Linen to Equinox.
  
* [PureJaxRL](https://github.com/luchris429/purejaxrl)

  * [**dqn_eqx.py**](./ports/purejaxrl/dqn_eqx.py): Port of the PureJaxRL DQN implementation[dqn.py](https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/dqn.py) from Flax/Linen to Equinox.

  * [**ppo_eqx.py**](./ports/purejaxrl/ppo_eqx.py): Port of the PureJaxRL PPO implementation[ppo.py](https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/ppo.py) from Flax/Linen to Equinox.

  * [**ppo_minigrid_eqx.py**](./ports/purejaxrl/ppo_minigrid_eqx.py): Port of the PureJaxRL PPO minigrid implementation[ppo_minigrid.py](https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/ppo_minigrid.py) from Flax/Linen to Equinox.

* [RLax](https://github.com/google-deepmind/rlax)
  * [**online_q_lambda_eqx.py**](./ports/rlax/online_q_lambda_eqx.py): Port of the RLax example [online_q_lambda.py](https://github.com/google-deepmind/rlax/blob/master/examples/online_q_lambda.py) from Haiku to Equinox.
  * [**online_q_learning_eqx.py**](./ports/rlax/online_q_learning_eqx.py): Port of the RLax example [online_q_learning.py](https://github.com/google-deepmind/rlax/blob/master/examples/online_q_learning.py) from Haiku to Equinox.
  * [**pop_art_eqx.py**](./ports/rlax/pop_art_eqx.py): Port of the RLax example [pop_art.py](https://github.com/google-deepmind/rlax/blob/master/examples/pop_art.py) from Haiku to Equinox.
  * [**simple_dqn_eqx.py**](./ports/rlax/simple_dqn_eqx.py): Port of the RLax example [simple_dqn.py](https://github.com/google-deepmind/rlax/blob/master/examples/simple_dqn.py) from Haiku to Equinox.

  
## Variants

* Each port has a corresponding `_orig` version in the repository, containing the original implementation for quick comparison.

* Some ports also include an `_opt` variant with performance optimizations based on [Low-overhead training loops](https://docs.kidger.site/equinox/tricks/#low-overhead-training-loops). While these optimizations offer some performance improvements, the gains are relatively modest. We hope these examples make it straightforward to apply these optimizations to the remaining ports as needed.
