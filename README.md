# PureEqxRL

A minimalist reinforcement learning implementation in JAX, powered by [Equinox](https://github.com/patrick-kidger/equinox). This project is a port of [PureJaxRL](https://github.com/luchris429/purejaxrl), replacing the [Flax-Linen](https://flax-linen.readthedocs.io/en/latest/) neural network implementation with [Equinox](https://github.com/patrick-kidger/equinox).

## Project Status?

This project is in very early development. Currently we have:

* Ported the [DQN](https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/dqn.py) (Deep Q-Network) implementation from [PureJaxRL](https://github.com/luchris429/purejaxrl)
* Initial performance benchmarking between Equinox and Flax based implementations
* Found that our Equinox-based version currently runs slower than the original Flax version.

### Next steps:

* Investigating performance bottlenecks
* Planning ports of additional algorithms from PureJaxRL
* Setting up systematic benchmarking infrastructure


## Getting Started

```bash
git clone https://github.com/yourusername/pureeqxrl
cd pureeqxrl
pip install -r requirements.txt
cd pureeqxrl
python dqn_eqx.py
```
We also added the original PureJaxRL implementation (see [dqn_flax_linen](./pureeqxrl/dqn_flax_linen.py))



## License

This project is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for details.

PureEqxRL is a port of [PureJaxRL](https://github.com/luchris429/purejaxrl), which is also licensed under the Apache License 2.0. The copyright for [PureJaxRL](https://github.com/luchris429/purejaxrl) belongs to its respective contributors.

## Contributing
We welcome contributions, especially in the areas of:

* Performance optimization
* Additional RL algorithm implementations
* Bug fixes and testing

## Acknowledgements

Special thanks to [luchris429](https://github.com/luchris429) for creating the original [PureJaxRL](https://github.com/luchris429/purejaxrl) repository.


## Changelog

### 2024.12.23
- Initial commit.