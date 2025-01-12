# PureEqxRL

A minimalist reinforcement learning implementation in JAX, powered by [Equinox](https://github.com/patrick-kidger/equinox). This project aims to reimplement [PureJaxRL](https://github.com/luchris429/purejaxrl), replacing the [Flax-Linen](https://flax-linen.readthedocs.io/en/latest/) neural network implementation with [Equinox](https://github.com/patrick-kidger/equinox).

## Project Status

This project is in very early development. So far we have only ported the [DQN](https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/dqn.py) (Deep Q-Network) implementation from [PureJaxRL](https://github.com/luchris429/purejaxrl). Our main focus at the moment is on comparing performance between the two implementations and improving our Equinox-based version, which shows lower performance.


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

| Metric                    | Linen (ms)    | Equinox (ms)  | Difference (%) |
|--------------------------|---------------|----------------|--------------|
| Full step                | 0.183         | 0.234         | 28.16       |
| Forward pass             | 0.027         | 0.049         | 80.01       |
| Experience collection     | 0.089         | 0.132         | 48.59       |
| Policy update            | 0.086         | 0.121         | 41.06       |

Based on the performance recommendations provided in the issue [Equinox much slower than flax linen when passing pytrees](https://github.com/patrick-kidger/equinox/issues/928), we achieved the following improvements (for details, see [simple_benchmark_flatten.py](./benchmarks/simple_benchmark_flatten.py))

| Metric                    | Linen (ms)          | Equinox (ms)        | Difference (%)  |
|-------------------------|---------------|---------------|---------------|
| Full step            | 0.189           | 0.221           | 16.78           |
| Forward pass             | 0.028           | 0.038           | 33.62           |
| Experience collection              | 0.090           | 0.109           | 20.98           |
| Policy update        | 0.086           | 0.104           | 20.20           |

We will look further into this ...

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