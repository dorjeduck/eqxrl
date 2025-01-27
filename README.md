# EqxRL

This repository contains implementations of various reinforcement learning algorithms ported to [JAX](https://github.com/jax-ml/jax) and [Equinox](https://github.com/patrick-kidger/equinox). The main goal is to learn and understand both RL concepts and JAX/Equinox frameworks through hands-on implementation, prioritizing clarity over performance optimizations for now.

> **Note**: While most JAX-based RL implementations use Flax/Linen for neural network components, this project specifically explores using Equinox as an alternative. This is mainly for learning purposes and to understand the differences between these libraries.


## Implementations

This is an active work in progress, with frequent updates expected. The goal is to gradually incorporate additional reinforcement learning algorithms to this repository.

* [**dqn_eqx.py**](./pureeqxrl/dqn_eqx.py): Direct port of the [PureJaxRL DQN implementation](https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/dqn.py) from Flax/Linen to Equinox. (optimized version: [**dqn_eqx_flatten.py**](./pureeqxrl/dqn_eqx_flatten.py)) 

* [**ppo_eqx.py**](./pureeqxrl/ppo_eqx.py): Direct port of the [PureJaxRL PPO implementation](https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/ppo.py) from Flax/Linen to Equinox. (optimized version: [**ppo_eqx_flatten.py**](./pureeqxrl/ppo_eqx_flatten.py)) 

The `_flatten` implementations are performance optimized based on [Low-overhead training loops](https://docs.kidger.site/equinox/tricks/#low-overhead-training-loops).


## Changelog

- 2025.01.27
    - PureJaxRL PPO port added.
- 2024.12.28
    - PureJaxRL DQN port added.
- 2024.12.23
    - Initial commit.


## License

This project is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for details.