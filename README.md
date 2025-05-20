# EqxRL

This repository contains implementations of various reinforcement learning algorithms ported to [JAX](https://github.com/jax-ml/jax)/[Equinox](https://github.com/patrick-kidger/equinox). The goal is to learn Deep RL and JAX/Equinox through clear, hands-on implementations, with clarity prioritized over performance.

> **Note**: While most JAX-based RL implementations use Flax/Linen for neural network components, this project specifically explores using Equinox as an alternative.

## Implementations

This is an ongoing work in progress. The goal is to gradually incorporate most common reinforcement learning algorithms to this repository.

### Ports

See [ports](./ports.md) for a list ports of Reinforcement Learning implementations to Jax/Equinox we have created so far.

## Changelog

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
