# EqxRL

This repository contains implementations of various reinforcement learning algorithms ported to [JAX](https://github.com/jax-ml/jax) and [Equinox](https://github.com/patrick-kidger/equinox). The main goal is to learn and understand both Deep Reinforcement Learning concepts and the JAX/Equinox frameworks through hands-on implementation, prioritizing clarity over performance optimizations for now.

> **Note**: While most JAX-based RL implementations use Flax/Linen for neural network components, this project specifically explores using Equinox as an alternative.

## Implementations

This is an ongoing work in progress. The goal is to gradually incorporate additional reinforcement learning algorithms to this repository.

* [CleanRL](https://github.com/vwxyzjn/cleanrl)

  * [**dqn_jax_eqx.py**](./ports/cleanrl/dqn_jax_eqx.py): Port of the CleanRL DQN implementation [dqn_jax.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_jax.py) from Flax/Linen to Equinox. (optimized version: [**dqn_jax_eqx_flatten.py**](./ports/cleanrl/dqn_jax_eqx_flatten.py))
  
* [PureJaxRL](https://github.com/luchris429/purejaxrl)

  * [**dqn_eqx.py**](./ports/purejaxrl/dqn_eqx.py): Port of the PureJaxRL DQN implementation[dqn.py](https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/dqn.py) from Flax/Linen to Equinox. (optimized version: [**dqn_eqx_flatten.py**](./ports/purejaxrl/dqn_eqx_flatten.py))

  * [**ppo_eqx.py**](./ports/purejaxrl/ppo_eqx.py): Port of the PureJaxRL PPO implementation[ppo.py](https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/ppo.py) from Flax/Linen to Equinox. (optimized version: [**ppo_eqx_flatten.py**](./ports/purejaxrl/ppo_eqx_flatten.py))

  * [**ppo_minigrid_eqx.py**](./ports/purejaxrl/ppo_minigrid_eqx.py): Port of the PureJaxRL PPO minigrid implementation[ppo_minigrid.py](https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/ppo_minigrid.py) from Flax/Linen to Equinox. (optimized version: [**ppo_minigrid_eqx_flatten.py**](./ports/purejaxrl/ppo_minigrid_eqx_flatten.py))

The `_flatten` implementations are performance optimized based on [Low-overhead training loops](https://docs.kidger.site/equinox/tricks/#low-overhead-training-loops).

## Changelog

* 2025.05.13
  * CleanRL DQN port added. 
* 2025.05.08
  * PureJaxRL PPO minigrid port added.
* 2025.01.27
  * PureJaxRL PPO port added.
* 2024.12.28
  * PureJaxRL DQN port added.
* 2024.12.23
  * Initial commit.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for details.
