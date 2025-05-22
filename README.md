# EqxRL

This repository explores the implementation of reinforcement learning algorithms with [JAX](https://github.com/jax-ml/jax) & [Equinox](https://github.com/patrick-kidger/equinox). While most JAX-based RL implementations we are aware of use the now discontinued [Flax/Linen](https://flax-linen.readthedocs.io/en/latest/) and [Haiku](https://github.com/google-deepmind/dm-haiku) neural network libraries, this project specifically explores using Equinox as an alternative.

## Project Overview

Currently, we're reimplementing popular reinforcement learning algorithms using [Equinox](https://github.com/patrick-kidger/equinox) to evaluate its suitability for RL workloads.

As a possible next step, we're considering a lightweight library built on top of [RLax](https://github.com/google-deepmind/rlax), specifically tailored for Equinox-based reinforcement learning workflows.

### Ports

See [ports](./ports.md) for a list of ports of Reinforcement Learning implementations to Jax/Equinox we have implemented so far.

## Changelog

* 2025.05.22
  * BSuite example ports added.
* 2025.05.20
  * RLax example ports added.
* 2025.05.17
  * CleanRL Atari DQN port added.
* 2025.05.16
  * CleanRL Twin Delayed Deep Deterministic Policy Gradient port added.
* 2025.05.15
  * CleanRL DDPQ port added.
* 2025.05.13
  * CleanRL DQN & Categorical DQN port added.
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
