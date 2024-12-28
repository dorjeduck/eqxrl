# PureEqxRL

A minimalist reinforcement learning implementation in JAX, powered by [Equinox](https://github.com/patrick-kidger/equinox). This project is a port of [PureJaxRL](https://github.com/luchris429/purejaxrl), replacing the [Flax-Linen](https://flax-linen.readthedocs.io/en/latest/) neural network implementation with [Equinox](https://github.com/patrick-kidger/equinox).

## Project Status?

This project is in very early development. Currently we have:

* Ported the [DQN](https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/dqn.py) (Deep Q-Network) implementation from [PureJaxRL](https://github.com/luchris429/purejaxrl)
* Set up initial performance benchmarking to compare Equinox and Flax implementations (see Benchmarking section below)
* Found our Equinox port currently runs slower than the original Flax implementation, prompting ongoing optimization work.


## Getting Started

```bash
git clone https://github.com/yourusername/pureeqxrl
cd pureeqxrl
pip install -r requirements.txt
cd pureeqxrl
python dqn_eqx.py
```
We also added the original PureJaxRL implementation (see [dqn_flax_linen.py](./pureeqxrl/dqn_flax_linen.py))

## Benchmarking

To better understand the performance differences between Flax/Linen and Equinox implementations, we started looking into potential bottlenecks. For this investigation, we set up a simple benchmark [simple_benchmark.py](./benchmarks/simple_benchmark.py) that breaks down key operations in the DQN algorithm:

| Metric                    | Linen (ms)    | Equinox (ms)  | Slowdown (%) |
|--------------------------|---------------|----------------|--------------|
| Full step                | 0.183         | 0.234         | 28.16       |
| Forward pass             | 0.027         | 0.049         | 80.01       |
| Experience collection     | 0.089         | 0.132         | 48.59       |
| Policy update            | 0.086         | 0.121         | 41.06       |

The initial results show that our Equinox implementation is running slower across all measured operations, with the forward pass showing the largest performance gap. This data provides a starting point for targeted optimization efforts.

### Next steps:

* Further investigating performance bottlenecks
* Planning ports of additional algorithms from PureJaxRL
* Setting up a systematic benchmarking infrastructure


## Contributing
We welcome contributions, especially in the areas of:

* Performance optimization
* Additional RL algorithm implementations
* Bug fixes and testing


## Acknowledgements

Special thanks to [luchris429](https://github.com/luchris429) for creating the original [PureJaxRL](https://github.com/luchris429/purejaxrl) repository and to [patrick-kidger](https://github.com/patrick-kidger) for the awesome [Equinox](https://github.com/patrick-kidger/equinox) library. 


## Changelog

- 2024.12.28
    - Simple benchmark added.
- 2024.12.23
    - Initial commit.


## License

This project is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for details.

PureEqxRL is a port of [PureJaxRL](https://github.com/luchris429/purejaxrl), which is also licensed under the Apache License 2.0. The copyright for [PureJaxRL](https://github.com/luchris429/purejaxrl) belongs to its respective contributors.