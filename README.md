# EqxRL

⚠️ Work In Progress ⚠️

This repository contains implementations of various reinforcement learning algorithms ported to [JAX](https://github.com/jax-ml/jax) and [Equinox](https://github.com/patrick-kidger/equinox). The main goal is to learn and understand both RL concepts and JAX/Equinox frameworks through hands-on implementation. 


> **Note**: While most JAX-based RL implementations use Flax/Linen for neural network components, this project specifically explores using Equinox as an alternative. This is mainly for learning purposes and to understand the differences between these approaches.

## Overview
This project focuses on reimplementing classic RL algorithms using JAX's functional programming paradigm and Equinox's neural network utilities. By porting existing implementations to JAX/Equinox, we can leverage:

* JAX's automatic differentiation
* Hardware acceleration (GPU/TPU support)
* Just-in-time compilation
* Functional transformations (vmap, pmap)

## Current Status
This is an active work in progress. Expect frequent changes, refactoring, and potential breaking changes. Some implementations may be incomplete or unoptimized.

## Implementations

* [./pureeqxrl/dqn_eqx.py](./pureeqxrl/dqn_eqx.py) Direct port of [PureJaxRl DQN implementation](https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/dqn.py) from Flax/Linen to Equinox. 
* [./pureeqxrl/dqn_eqx_flatten.py](./pureeqxrl/dqn_eqx_flatten.py) Performance-optimized version of [./pureeqxrl/dqn_eqx.py], based on [Low-overhead training loops](https://docs.kidger.site/equinox/tricks/#low-overhead-training-loops). (marginal performance gain)

more to come ...


## Notes
This is a personal learning project. Implementations may not be production-ready and are primarily focused on understanding RL concepts and JAX/Equinox programming patterns.


## Changelog

- 2024.12.28
    - Simple benchmark added.
- 2024.12.23
    - Initial commit.


## License

This project is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for details.